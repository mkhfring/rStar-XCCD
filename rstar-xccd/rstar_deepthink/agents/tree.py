# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations
import os
import re
from abc import abstractmethod
from termcolor import colored
from typing import Optional, Any, Dict, List, Callable, Type, Tuple, Union
from pydantic import BaseModel, PrivateAttr, conlist, ConfigDict, field_validator
from omegaconf import DictConfig, OmegaConf
from timeout_decorator import timeout
from rstar_deepthink.config import BaseConfig
from rstar_deepthink.nodes.base_node import BaseNode
from rstar_deepthink.tools.python_tool import PythonInterpreter, extract_code1_python, sanitize_input, is_python_code, is_java_execution_attempt
from rstar_deepthink.constants import TIMEOUT_SECONDS, TIMEOUT_MESSAGE, CODE_END, OUTPUT_END, CODE, ANSWER


def _python_ast_init():
    python = PythonInterpreter(globals=globals(), locals=None)
    return python


def tool_wrapper(tool):
    def _tool(query):
        return tool.run(query)
    return _tool


def no_action_wrapper(tool):
    def _tool(query):
        return True, "No action, no observation. Please continue to solve."
    return _tool


# How a python_interpreter step turned out. The scorer in mcts.py has to
# tell these apart -- above all, whether any code actually ran -- and used
# to infer it from the observation text, which cannot distinguish a sentinel
# message from a successful run (see MCTS_SEARCH_SCORING.md).
EXEC_OK = "ok"                              # ran to completion, nothing raised
EXEC_ERROR = "error"                        # raised a non-assertion exception
EXEC_ASSERTION_FAILED = "assertion_failed"  # raised AssertionError
EXEC_NO_CODE = "no_code"                    # no runnable Python was produced
EXEC_BLOCKED = "blocked"                    # tried to execute the non-Python side

NO_CODE_MESSAGE = "No valid Python code found in the response."
JAVA_BLOCKED_MESSAGE = (
    "Java execution is not supported in this sandbox (no JDK on PATH). "
    "Do not import java.* or shell out to java/javac; "
    "reason about Code 2 (Java) by static analysis instead."
)


tools = {
    "python_interpreter": tool_wrapper(_python_ast_init()),
    "None": no_action_wrapper(_python_ast_init()),
}


class BaseTree(BaseModel):

    config: Any
    question: str
    ground_truth: Optional[Union[str, List[str]]] = None
    llm: Any = None
    root: Optional[Type[BaseNode]] = None
    current_node: Optional[Type[BaseNode]] = None 
    stop: Optional[List[str]] = None
    node_max_retry: int = 5
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if self.config.stop:
            # omegaconf.listconfig.ListConfig -> list
            self.stop = OmegaConf.to_object(self.config.stop)

        self.root = self.create_root()
        self.current_node = self.root
    
    @field_validator("config")
    def validate_config(cls, cfg: Any):
        if issubclass(type(cfg), DictConfig):
            if not os.path.exists(cfg.model_dir):
                raise ValueError(f"Model directory \"{cfg.model_dir}\" cannot be found.")
            return cfg

        raise TypeError("Wrong type for `config`, must be subclass of BaseConfig")
    
    def create_root(self) -> Type[BaseNode]:
        root = self.create_node()
        root.state["extra_info"] = f"question: {self.question}"
        return root

    @abstractmethod
    def create_node(self, parent: Optional[Type[BaseNode]] = None) -> Type[BaseNode]:
        """
        subclass must implement
        """
    
    def collect_partial_solution(self, node: Type[BaseNode]) -> str:
        # from leaf to root, and reverse
        trajectory = []
        while node:
            if node.state['text']:
                trajectory.append(node.state['text'])
            node = node.parent
        return "".join(reversed(trajectory))
    
    def return_states(self) -> Dict[str, Dict[str, str]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            if node.has_children():
                candidates.extend(node.children)
        return states
    
FENCE_LINE_RE = re.compile(r"^\s*`{3,}\s*\w*\s*$")
# Non-greedy, DOTALL so a block can span multiple lines or share a line with
# its own tags (e.g. "<code> print(x) <end_of_code>"). Falls back to the end
# of the string if <end_of_code> is missing (e.g. truncated generation).
CODE_BLOCK_RE = re.compile(r"<code>(.*?)(?:<end_of_code>|$)", re.DOTALL)


def _clean_code_block(block: str) -> str:
    lines = [line for line in block.split("\n") if not FENCE_LINE_RE.match(line)]
    return "\n".join(lines).strip()


def extract_program(result: str, last_only=False):
    result = result.replace("<end_of_step>", "")
    blocks = CODE_BLOCK_RE.findall(result)
    if not blocks:
        return ""
    if last_only:
        return _clean_code_block(blocks[-1])

    program = _clean_code_block(blocks[0])
    for block in blocks[1:]:
        # only separate from a previously extracted code block,
        # not before the first one
        program += "\n# ========\n" + _clean_code_block(block)
    return program.strip()

def _get_root_question(node: Type[BaseNode]) -> str:
    """Walk up to the root node and return the question string from extra_info."""
    while node.parent is not None:
        node = node.parent
    extra_info = node.state.get("extra_info", "")
    prefix = "question: "
    if extra_info.startswith(prefix):
        return extra_info[len(prefix):]
    return extra_info


def code_execution(
    node: Type[BaseNode],
    parser_result: Dict[str, str],
) -> Tuple[str, str]:
    """Execute a step's code, returning (observation, outcome).

    `outcome` is one of the EXEC_* constants above. It is derived from the
    interpreter's own exception flag rather than from the observation text,
    so callers can distinguish "the code ran and printed something" from
    "nothing ran at all" -- a distinction the previous
    `"error" not in observation.lower()` test could not make.
    """

    @timeout(TIMEOUT_SECONDS, use_signals=True, exception_message=TIMEOUT_MESSAGE)
    def _code_execution(node: Type[BaseNode], parser_result: Dict[str, str]) -> Tuple[str, str]:
        # Define tool
        action = parser_result["action"]
        tool_func = tools[action]

        history_action_inputs = collect_action_inputs(node, action)

        # then, we execute current code snippets
        action_input = parser_result["action_input"]
        action_input = extract_program(''.join(history_action_inputs) + action_input)

        if action == "python_interpreter":
            sanitized_code = sanitize_input(action_input)
            if not is_python_code(sanitized_code):
                return NO_CODE_MESSAGE, EXEC_NO_CODE
            if is_java_execution_attempt(sanitized_code):
                return JAVA_BLOCKED_MESSAGE, EXEC_BLOCKED

        # Write Code 1 as candidate_code.py so subprocess tests can run it.
        question = _get_root_question(node)
        code1 = extract_code1_python(question)
        if code1:
            with open("candidate_code.py", "w") as f:
                f.write(code1)

        ran_without_raising, observation = tool_func(action_input)
        observation = str(observation).strip()
        del tool_func
        if ran_without_raising:
            outcome = EXEC_OK
        elif observation.startswith("AssertionError"):
            # Sub-classifying a genuine exception by the type name the
            # interpreter itself formatted -- not a guess about whether an
            # exception happened at all, which `ran_without_raising` already
            # settled.
            outcome = EXEC_ASSERTION_FAILED
        else:
            outcome = EXEC_ERROR
        return observation, outcome
    try:
        observation, outcome = _code_execution(node, parser_result)
    except Exception as e:
        # Anything escaping here (the timeout above included) is a failure.
        observation = "{}: {}".format(type(e).__name__, str(e))
        outcome = EXEC_ASSERTION_FAILED if isinstance(e, AssertionError) else EXEC_ERROR

    return observation, outcome


def collect_action_inputs(
    node: Type[BaseNode], 
    action: str,
) -> List[str]:
    action_inputs = []
    while node:
        if node.state["action"] == action:
            action_input = node.state["action_input"]
            if action_input and "TimeoutError" not in node.state["text"].split(action_input)[-1]:
                action_inputs.append(action_input)
        if OUTPUT_END in node.state['text'] or CODE_END in node.state['text']:
            break
        node = node.parent
    return action_inputs[::-1]


def code_run(solution):
    if CODE not in solution or CODE_END not in solution or OUTPUT_END not in solution or ANSWER not in solution:
        return solution
    
    @timeout(TIMEOUT_SECONDS, use_signals=True, exception_message=TIMEOUT_MESSAGE)
    def _code_execution(solution: str) -> str:
        tool_func = tools['python_interpreter']
        action_input = extract_program(solution)
        _ran_without_raising, observation = tool_func(action_input)
        observation = str(observation).strip()
        del tool_func
        return observation
    
    try:
        observation = _code_execution(solution)
    except Exception as e:
        observation = "{}: {}".format(type(e).__name__, str(e))
    
    return observation