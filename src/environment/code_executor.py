"""
Sandboxed Python code executor.
Runs agent-generated code against a problem's test cases and returns pass@1.
"""

import re
import resource
import subprocess
import sys
import tempfile
import textwrap
from typing import List, Dict, Optional


def _extract_helper_context(degraded_prompt: str, entry_point: str) -> str:
    """
    Extract helper functions and imports from the degraded prompt that the
    agent's code may depend on (e.g., poly() for find_zero).

    Returns everything from the degraded prompt EXCEPT the main function
    (identified by entry_point or 'candidate').
    """
    lines = degraded_prompt.split("\n")
    context_lines = []
    skip = False

    for line in lines:
        stripped = line.strip()

        # Detect start of the main function → start skipping
        if re.match(rf'def\s+({re.escape(entry_point)}|candidate)\s*\(', stripped):
            skip = True
            continue

        if skip:
            # Detect start of a different top-level function → stop skipping
            if re.match(r'def\s+\w+\s*\(', stripped) and not line[0].isspace():
                skip = False
                context_lines.append(line)
            # Otherwise still inside the main function body → skip
            continue

        context_lines.append(line)

    return "\n".join(context_lines).strip()


def _alias_main_function(code: str, entry_point: str) -> str:
    """
    Add an alias so the last top-level function in the code is accessible
    via entry_point name. This avoids renaming the wrong function when
    helper functions are defined before the main function.
    """
    # If the code already defines a function with the entry_point name, no alias needed
    if re.search(rf'def\s+{re.escape(entry_point)}\s*\(', code):
        return code

    # Find all top-level function names (not indented)
    func_names = re.findall(r'^def\s+(\w+)\s*\(', code, re.MULTILINE)
    if not func_names:
        return code

    # Alias the last defined function to entry_point
    last_func = func_names[-1]
    if last_func != entry_point:
        code = code.rstrip() + f"\n\n{entry_point} = {last_func}\n"

    return code


def _format_output(out) -> str:
    """
    Format a test case output value for insertion into an assert statement.
    Numeric and list outputs can be inserted directly; string outputs need quoting.
    """
    # If it's already a valid Python literal (number, list, bool, None), use as-is
    if isinstance(out, (int, float, bool, list, dict, type(None))):
        return repr(out)
    # String output — try to parse as a Python literal first
    s = str(out)
    try:
        import ast
        ast.literal_eval(s)
        # It's already a valid Python expression (e.g., "[1,2,3]", "True", "0.5")
        return s
    except (ValueError, SyntaxError):
        # It's a bare string (e.g., "fdcb") — needs quoting
        return repr(s)


def _expand_template_relation(rel: str, inp, entry_point: str) -> str:
    """
    Expand template-based test relations that use $demo$ and $input$ placeholders.

    - $demo$ imports are stripped (helper functions are included via context)
    - $input$ is replaced with the actual input value
    - candidate is replaced with entry_point
    - print(expr) is converted to assert expr
    """
    inp_str = str(inp)

    # Replace template variables
    expanded = rel.replace("$input$", inp_str)
    expanded = expanded.replace("candidate", entry_point)

    # Remove $demo$ import lines (the functions are included via context)
    result_lines = []
    for line in expanded.split("\n"):
        if "$demo$" in line:
            continue
        # Convert print(condition) to assert condition
        if line.strip().startswith("print(") and line.strip().endswith(")"):
            inner = line.strip()[6:-1]
            result_lines.append(f"assert {inner}")
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


# ── Test case → assert statement conversion ───────────────────────────────────

def build_test_program(
    code: str, test_cases: List[Dict], entry_point: str,
    context: str = "",
) -> str:
    """
    Build a self-contained Python program from the agent's code + test cases.

    Args:
        code:        the agent's extracted code
        test_cases:  list of test case dicts
        entry_point: expected function name for test assertions
        context:     helper functions/imports from the degraded spec (e.g., poly)
    """
    code = _alias_main_function(code, entry_point)
    lines = []

    # Include helper context (e.g., poly function, imports) before agent's code
    if context.strip():
        lines.append(textwrap.dedent(context))
        lines.append("")

    lines.append(textwrap.dedent(code))
    lines.append("")

    for tc in test_cases:
        inp = tc.get("input", "")
        out = tc.get("output", "")
        rel = tc.get("relation", "==")

        if rel == "==":
            out_expr = _format_output(out)
            lines.append(f"assert {entry_point}({inp}) == {out_expr}")
        elif "$demo$" in rel or "$input$" in rel:
            expr = _expand_template_relation(rel, inp, entry_point)
            lines.append(expr)
        elif "candidate" in rel:
            expr = rel.replace("candidate", entry_point)
            lines.append(f"assert {expr}")
        else:
            lines.append(f"assert {rel}")

    return "\n".join(lines)


def _build_single_test(
    code: str, test_case: Dict, entry_point: str,
    context: str = "",
) -> str:
    """Build a program that runs exactly one test case."""
    inp = test_case.get("input", "")
    out = test_case.get("output", "")
    rel = test_case.get("relation", "==")

    code = _alias_main_function(code, entry_point)
    lines = []

    if context.strip():
        lines.append(textwrap.dedent(context))
        lines.append("")

    lines.append(textwrap.dedent(code))
    lines.append("")

    if rel == "==":
        out_expr = _format_output(out)
        lines.append(f"assert {entry_point}({inp}) == {out_expr}")
    elif "$demo$" in rel or "$input$" in rel:
        expr = _expand_template_relation(rel, inp, entry_point)
        lines.append(expr)
    elif "candidate" in rel:
        expr = rel.replace("candidate", entry_point)
        lines.append(f"assert {expr}")
    else:
        lines.append(f"assert {rel}")

    return "\n".join(lines)


# ── Executor ──────────────────────────────────────────────────────────────────

class CodeExecutor:
    def __init__(self, cfg):
        self.timeout = cfg.code_executor.timeout
        self.partial_credit = cfg.code_executor.partial_credit

    def run(
        self, code: str, test_cases: List[Dict], entry_point: str,
        context: str = "",
    ) -> float:
        """
        Execute agent code against the test suite.

        Args:
            code:        the agent's extracted code
            test_cases:  test case dicts
            entry_point: expected function name
            context:     helper functions from the degraded spec (optional)

        Returns:
            pass@1 score in [0.0, 1.0]
        """
        if not code.strip():
            return 0.0
        if not test_cases:
            return 0.0

        if self.partial_credit:
            return self._run_partial(code, test_cases, entry_point, context)
        else:
            return self._run_all(code, test_cases, entry_point, context)

    def _run_all(self, code, test_cases, entry_point, context=""):
        program = build_test_program(code, test_cases, entry_point, context)
        success = _execute_program(program, self.timeout)
        return 1.0 if success else 0.0

    def _run_partial(self, code, test_cases, entry_point, context=""):
        if not test_cases:
            return 0.0

        passed = _execute_program_partial(code, test_cases, entry_point, context, self.timeout)
        return passed / len(test_cases)


def _build_partial_program(
    code: str, test_cases: List[Dict], entry_point: str, context: str = ""
) -> str:
    """
    Build a program that runs all test cases with try/except per test and
    prints the number of passing tests to stdout on the last line.
    Avoids spawning one subprocess per test case.
    """
    code = _alias_main_function(code, entry_point)
    lines = []

    if context.strip():
        lines.append(textwrap.dedent(context))
        lines.append("")

    lines.append(textwrap.dedent(code))
    lines.append("")
    lines.append("_passed = 0")
    lines.append("_total = 0")

    for tc in test_cases:
        inp = tc.get("input", "")
        out = tc.get("output", "")
        rel = tc.get("relation", "==")

        if rel == "==":
            out_expr = _format_output(out)
            assert_line = f"assert {entry_point}({inp}) == {out_expr}"
        elif "$demo$" in rel or "$input$" in rel:
            assert_line = _expand_template_relation(rel, inp, entry_point)
        elif "candidate" in rel:
            expr = rel.replace("candidate", entry_point)
            assert_line = f"assert {expr}"
        else:
            assert_line = f"assert {rel}"

        lines.append("_total += 1")
        lines.append("try:")
        for sub_line in assert_line.splitlines():
            lines.append(f"    {sub_line}")
        lines.append("    _passed += 1")
        lines.append("except Exception:")
        lines.append("    pass")

    lines.append("print(_passed)")
    return "\n".join(lines)


def _execute_program_partial(
    code: str, test_cases: List[Dict], entry_point: str,
    context: str, timeout: float,
) -> int:
    """
    Run all test cases in a single subprocess.
    Returns the number of tests that passed (int).
    """
    program = _build_partial_program(code, test_cases, entry_point, context)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(program)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        if result.returncode == 0:
            try:
                return int(result.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                return 0
        return 0
    except subprocess.TimeoutExpired:
        return 0
    except Exception:
        return 0
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _execute_program(program: str, timeout: float) -> bool:
    """
    Run a Python program in a subprocess with timeout.
    Returns True if it exits with code 0, False otherwise.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(program)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Code extraction ───────────────────────────────────────────────────────────

def extract_code(action_text: str) -> str:
    """
    Extract Python code from the agent's [ANSWER] response.
    Handles:
      - Raw code after [ANSWER] tag
      - Markdown code blocks (```python ... ```)
    """
    # Strip [ANSWER] prefix
    text = action_text
    if text.upper().startswith("[ANSWER]"):
        text = text[len("[ANSWER]"):].strip()

    # Extract from markdown code block if present
    if "```" in text:
        start = text.find("```")
        # Skip language identifier line (```python)
        newline = text.find("\n", start)
        end = text.find("```", newline)
        if newline != -1 and end != -1:
            return text[newline + 1: end].strip()

    return text.strip()
