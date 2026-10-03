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
from rstar_deepthink.tools.python_tool import PythonInterpreter, extract_code1_python, extract_code2_source, stage_code2, mentions_code2_execution, sanitize_input, is_python_code, snapshot_dir, cleanup_generated_code, execute_code2_enabled, CODE2_DISABLED_MESSAGE, auto_code2_enabled, capture_python_inputs, auto_compare_code2, codestep_enabled, run_codestep
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
        return "No action, no observation. Please continue to solve."
    return _tool


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
# FIX 2026-09-30: a block also ends at the next <code>. When the model reopens
# <code> in a continuation step ("<code> def run_both..." then "<code> out =
# run_both(...) <end_of_code>"), the old pattern kept the literal second
# "<code>" inside the program, which then failed to parse ("No valid Python code
# found"): 29% of code steps in the E5 rust dual-exec run and 41% in the
# hard-negative rust dual runs; 88-98% of those parse after the fix. Python-only
# runs were barely affected (~1% of their rejections).
CODE_BLOCK_RE = re.compile(r"<code>(.*?)(?=<code>|<end_of_code>|$)", re.DOTALL)


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
) -> str:


    @timeout(TIMEOUT_SECONDS, use_signals=True, exception_message=TIMEOUT_MESSAGE)
    def _code_execution(node: Type[BaseNode], parser_result: Dict[str, str]) -> str:
        # Define tool
        action = parser_result["action"]
        if action == "python_interpreter" and codestep_enabled():
            # CODE-STEP MODE: every <code> block is one step; replay this trajectory's
            # earlier code steps (root -> parent) in a fresh namespace, then run this one.
            prev, a = [], node
            while a is not None:
                if a.state.get("action") == "python_interpreter" and a.state.get("action_input"):
                    prev.append(extract_program(a.state["action_input"]))
                a = a.parent
            current = extract_program(parser_result["action_input"])
            if not is_python_code(sanitize_input(current)):
                return "No valid Python code found in the response."
            return run_codestep(_get_root_question(node), [p for p in prev[::-1] if p], current)
        tool_func = tools[action]

        history_action_inputs = collect_action_inputs(node, action)

        # then, we execute current code snippets
        action_input = parser_result["action_input"]
        action_input = extract_program(''.join(history_action_inputs) + action_input)

        if action == "python_interpreter":
            sanitized_code = sanitize_input(action_input)
            if not is_python_code(sanitized_code):
                return "No valid Python code found in the response."

        question = _get_root_question(node)

        # Taken before anything is staged so the cleanup below also reclaims
        # candidate_code.py and the Code 2 artifacts, not just whatever the
        # model's own test code writes.
        before_files = snapshot_dir()

        # Write Code 1 as candidate_code.py so subprocess tests can run it.
        code1 = extract_code1_python(question)
        if code1:
            with open("candidate_code.py", "w") as f:
                f.write(code1)

        # Only stage/compile Code 2 (javac/java, rustc are on PATH on this
        # cluster) when the step actually looks like it's trying to run it --
        # avoids a wasted compile subprocess on every one of the many
        # Python-only test steps.
        code2_status = ""
        if action == "python_interpreter" and mentions_code2_execution(sanitized_code):
            code2_lang, code2_source = extract_code2_source(question)
            if code2_source:
                code2_status = (stage_code2(code2_lang, code2_source) if execute_code2_enabled()
                                else CODE2_DISABLED_MESSAGE)

        try:
            if action == "python_interpreter" and auto_code2_enabled() and execute_code2_enabled():
                with capture_python_inputs() as cap:
                    observation = str(tool_func(action_input)).strip()
                # before the cleanup below: auto_compare_code2 stages its own files
                auto = auto_compare_code2(question, cap.inputs)
                if auto:
                    observation = f"{observation}\n{auto}"
            else:
                observation = str(tool_func(action_input)).strip()
        finally:
            # in a finally so a timeout or a raising step still cleans up;
            # every step re-stages what it needs, so nothing downstream
            # depends on these files surviving.
            cleanup_generated_code(before_files)
        del tool_func
        if code2_status:
            observation = f"{observation}\n{code2_status}"
        return observation
    try:
        observation = _code_execution(node, parser_result)
    except Exception as e:
        observation = "{}: {}".format(type(e).__name__, str(e))
    
    return observation


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
        before_files = snapshot_dir()
        try:
            observation = str(tool_func(action_input)).strip()
        finally:
            # Same reason as in code_execution(): the re-run of a finished
            # solution writes the same scratch files, and they would
            # otherwise accumulate in the repository.
            cleanup_generated_code(before_files)
        del tool_func
        return observation
    
    try:
        observation = _code_execution(solution)
    except Exception as e:
        observation = "{}: {}".format(type(e).__name__, str(e))
    
    return observation