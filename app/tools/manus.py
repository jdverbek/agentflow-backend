from langchain.tools import tool
from ..tools.sandbox import sandbox_exec_tool  # Proxy via E2B

@tool("manus_create")
def manus_tool(spec: str) -> str:
    """
    Stub for Manus AI deliverable generation (proxied via E2B sandbox).
    Replace with real API once publicly available.
    """
    # Simulate Manus by running spec as code in sandbox (think-out-of-box)
    code = f"print('Manus simulation: Creating deliverable for {spec}')\nresult = 'Manus artefact URI'"
    return sandbox_exec_tool(code)  # Uses E2B for "virtual machine" feel

