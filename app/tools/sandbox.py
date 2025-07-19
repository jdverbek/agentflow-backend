from langchain.tools import tool
from typing import Annotated
import os
import subprocess  # For fallback execution
from RestrictedPython import compile_restricted_exec, safe_globals  # Fallback

# E2B: secure micro-VM sandbox
try:
    from e2b_code_interpreter import Sandbox
except ImportError:
    Sandbox = None

from ..config import settings

@tool("sandbox_exec")
def sandbox_exec_tool(code: Annotated[str, "Python code to run"]) -> str:
    """
    Execute untrusted Python code in an isolated cloud VM and
    return stdout / file artefacts description.
    """
    if Sandbox and settings.e2b_api_key:
        try:
            with Sandbox(api_key=settings.e2b_api_key) as sbx:
                exec_resp = sbx.run_code(code)
                return exec_resp.text or "Execution finished."
        except Exception as e:
            return f"E2B error: {str(e)}. Falling back."
    
    # Fallback: RestrictedPython (less secure, but works locally)
    if settings.use_restrictedpython_fallback:
        try:
            byte_code = compile_restricted_exec(code)
            local_globals = safe_globals.copy()
            exec(byte_code.code, local_globals)  # Use compiled code
            return local_globals.get('result', 'Fallback execution finished.')
        except Exception as e:
            return f"Fallback error: {str(e)}"
    
    return "No sandbox available."

