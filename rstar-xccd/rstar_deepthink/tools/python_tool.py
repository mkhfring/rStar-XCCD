# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
import argparse
import ast
import re
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, Optional, Tuple, Type, List
from pydantic import BaseModel, Field, root_validator
from timeout_decorator import timeout


TIMEOUT_SECONDS = 30
TIMEOUT_MESSAGE = f"Execution of the code snippet has timed out for exceeding {TIMEOUT_SECONDS} seconds."

def truncate_string(text, max_length=1024, is_evalf=True):
    # print(text, file=sys.stderr)
    # print(type(text), file=sys.stderr)
    if is_evalf and isinstance(text, str):
        try:
            text_sympy = float(text)
            refine_text = str(round(text_sympy, 4)) + "\n"
            text = text if len(text) < len(refine_text) else refine_text
        except:
            pass

    if len(str(text)) > max_length:
        return str(text)[:max_length//2] + "..." + str(text)[-max_length//2:]
    return text

def extract_content(text):
    pattern = r'print\((.*?)\)'
    matches = re.findall(pattern, text)
    if len(matches) < 1:
        return ""
    return " ".join(matches)+":"


def __is_print_node(node: ast.AST) -> bool:
    
    if isinstance(node, ast.Expr) and \
        isinstance(node.value, ast.Call) and \
        isinstance(node.value.func, ast.Name) and \
        node.value.func.id == "print":
        return True
    elif isinstance(node, ast.If) or \
         isinstance(node, ast.While) or \
         isinstance(node, ast.For) or \
         isinstance(node, ast.FunctionDef):
        for sub_node in node.body:
            if __is_print_node(sub_node):
                return True
    return False


def find_print_node(body: List[ast.AST]) -> List[int]:
    """Find the python print node in the tree.body.

    Args:
        body (List[ast.AST]): The body of the AST

    Returns:
        List[int]: The index of the python print node
    """
    print_index = []
    for idx, node in enumerate(body):
        if __is_print_node(node):
            print_index.append(idx)
    return print_index


def sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.
    Remove whitespace, backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """

    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    return query


def is_python_code(query: str) -> bool:
    """Check whether a code snippet parses as valid, non-empty Python.

    Used to gate execution: prose, Java snippets, or empty extractions
    should not be handed to the interpreter.
    """
    try:
        tree = ast.parse(query)
    except SyntaxError:
        return False
    return len(tree.body) > 0


JAVA_TOKEN_RE = re.compile(r"\bjavac?\b")


def is_java_execution_attempt(query: str) -> bool:
    """Check whether a python_interpreter query tries to touch Java.

    `import java.util.Scanner;` is syntactically valid Python (a dotted
    import), and `subprocess.run(["java", ...])` is itself valid Python, so
    neither is caught by is_python_code. There is no JDK on PATH in this
    sandbox, so both always fail at runtime (ModuleNotFoundError /
    FileNotFoundError). Catch them ahead of execution instead.
    """
    try:
        tree = ast.parse(query)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in ("java", "javax") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in ("java", "javax"):
                return True
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if JAVA_TOKEN_RE.search(node.value):
                return True
    return False


def extract_code1_python(question: str) -> str:
    """Extract the Python source of Code 1 from a clone-detection question.

    Looks for the block introduced by 'Code 1: Python\\n```python\\n' and
    returns everything up to the closing '```'.  Returns an empty string if
    the pattern is not found.
    """
    marker = "Code 1: Python\n```python\n"
    start = question.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    end = question.find("\n```", start)
    if end == -1:
        return ""
    return question[start:end]


class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")
    

class PythonInterpreter(BaseModel):
    """A tool for running python code snippet."""

    name: str = "python_interpreter"
    description: str = (
        "A Python shell. Use this to execute python commands. "
    )
    description_zh: str = (
        "Python 交互式 shell。使用此工具来执行 Python 代码。"
    )
    globals: Optional[Dict] = Field(default_factory=dict)
    locals: Optional[Dict] = Field(default_factory=dict)
    sanitize_input: bool = True
    max_length: int = 1024
    is_evalf: bool = True
    args_schema: Type[BaseModel] = PythonInputs
    use_signals: bool = False 

    def _base_run(
        self,
        query: str,
    ) -> Tuple[bool, str]:
        """Run `query`, returning (nothing_was_raised, output_text).

        The boolean is the authoritative "did this code raise?" signal.
        Callers used to have to infer it by searching the output text for
        the substring "error", which misfires in both directions: a program
        that merely prints the word "error" as data looks like a crash, and
        a sentinel message that happens to contain no such word looks like a
        clean run.

        Note that `_sub_run`'s own flag is *not* an error signal -- it only
        reports whether the final statement was an expression (eval) or had
        to be exec'd, and the exec fallback is a normal, successful path.
        """
        def _sub_run(bodys):
            io_buffer = StringIO()
            module = ast.Module(bodys[:-1], type_ignores=[])
            exec(ast.unparse(module), self.globals, self.locals)  # type: ignore
            module_end = ast.Module(bodys[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore

            try:
                with redirect_stdout(io_buffer):
                    ret = eval(module_end_str, self.globals, self.locals)
                    if ret is None:
                        return True, truncate_string(io_buffer.getvalue(), max_length=self.max_length, is_evalf=self.is_evalf)
                    else:
                        return True, truncate_string(ret, max_length=self.max_length, is_evalf=self.is_evalf)
            except Exception:
                with redirect_stdout(io_buffer):
                    exec(module_end_str, self.globals, self.locals)
                return False, truncate_string(io_buffer.getvalue(), max_length=self.max_length, is_evalf=self.is_evalf)
            
        try:
            if self.sanitize_input:
                query = sanitize_input(query)
            tree = ast.parse(query)
            print_indexs = find_print_node(tree.body)
            if len(print_indexs) == 0:
                print_indexs = [len(tree.body) - 1]
            ret_strs = []
            if len(print_indexs) == 1:
                run_flag, ret = _sub_run(tree.body)
                return True, f"{ret}"
            for start_idx, end_idx in zip([-1] + print_indexs, print_indexs):
                node_source = ast.get_source_segment(query, tree.body[end_idx])
                run_flag, ret = _sub_run(tree.body[start_idx + 1:end_idx + 1])
                ret_strs.append(f"{extract_content(node_source)} {ret}")
            return True, "".join(ret_strs)
        except Exception as e:
            return False, "{}: {}".format(type(e).__name__, str(e))

    def run(
        self,
        query: str,
    ) -> Tuple[bool, str]:

        @timeout(TIMEOUT_SECONDS, use_signals=True, exception_message=TIMEOUT_MESSAGE)
        def base_run(query: str) -> Tuple[bool, str]:
            return self._base_run(query)

        try:
            return base_run(query)
        except Exception as e:
            print(e)
            print(" exec code error ")
            return False, "{}: {}".format(type(e).__name__, str(e))


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--testcase', type=str, default="```python\nprint(1)\nprint(2)\n```")
    # input args
    args = args.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    tool = PythonInterpreter()
    ran_without_raising, output = tool.run(args.testcase)
    print(output)
    sys.exit(0 if ran_without_raising else 1)