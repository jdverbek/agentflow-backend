"""
Improved sandbox execution tool with E2B integration and RestrictedPython fallback.
Integrates with existing AgentFlow tools while adding secure code execution.
"""
from crewai_tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os
import logging
from RestrictedPython import compile_restricted_exec, safe_globals

# E2B: secure micro-VM sandbox
try:
    from e2b_code_interpreter import Sandbox
    E2B_AVAILABLE = True
except ImportError:
    Sandbox = None
    E2B_AVAILABLE = False

from ..config.improved_settings import improved_settings

logger = logging.getLogger(__name__)

class SandboxInput(BaseModel):
    """Input schema for sandbox execution tool."""
    code: str = Field(..., description="Python code to run in secure sandbox")

class ImprovedSandboxTool(BaseTool):
    name: str = "improved_sandbox_exec"
    description: str = "Execute untrusted Python code in an isolated cloud VM (E2B) or fallback to RestrictedPython for secure local execution."
    args_schema: Type[BaseModel] = SandboxInput

    def _run(self, code: str) -> str:
        """Execute code in secure sandbox environment."""
        logger.info(f"Executing code in sandbox: {code[:100]}...")
        
        # Try E2B first (most secure)
        if E2B_AVAILABLE and Sandbox and improved_settings.e2b_api_key:
            try:
                logger.info("Using E2B sandbox for code execution")
                with Sandbox(api_key=improved_settings.e2b_api_key) as sbx:
                    exec_resp = sbx.run_code(code)
                    result = exec_resp.text or "E2B execution completed successfully."
                    logger.info("E2B execution successful")
                    return result
            except Exception as e:
                logger.warning(f"E2B execution failed: {str(e)}. Falling back to RestrictedPython.")
        
        # Fallback: RestrictedPython (less secure, but works locally)
        if improved_settings.use_restrictedpython_fallback:
            try:
                logger.info("Using RestrictedPython fallback for code execution")
                byte_code = compile_restricted_exec(code)
                if byte_code.errors:
                    return f"Code compilation errors: {byte_code.errors}"
                
                local_globals = safe_globals.copy()
                local_globals['__builtins__']['_print_'] = lambda *args: print(*args)
                local_globals['result'] = None
                
                exec(byte_code.code, local_globals)
                result = local_globals.get('result', 'RestrictedPython execution completed.')
                logger.info("RestrictedPython execution successful")
                return str(result)
            except Exception as e:
                error_msg = f"RestrictedPython execution error: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        error_msg = "No sandbox available for code execution."
        logger.error(error_msg)
        return error_msg

class ManusInput(BaseModel):
    """Input schema for Manus deliverable creation tool."""
    spec: str = Field(..., description="Specification for deliverable creation")

class ImprovedManusTool(BaseTool):
    name: str = "improved_manus_create"
    description: str = "Enhanced Manus AI deliverable generation tool. Uses E2B sandbox as proxy until Manus API becomes publicly available."
    args_schema: Type[BaseModel] = ManusInput

    def _run(self, spec: str) -> str:
        """Create deliverable using Manus simulation."""
        logger.info(f"Creating deliverable with spec: {spec[:100]}...")
        
        # Simulate Manus by running deliverable creation code in sandbox
        creation_code = f"""
# Manus simulation: Creating deliverable for {spec}
import json
import datetime

deliverable_info = {{
    'type': 'document',
    'specification': '{spec}',
    'created_at': datetime.datetime.now().isoformat(),
    'status': 'created',
    'uri': f'manus://deliverable/{{hash('{spec}')}}',
    'format': 'auto-detected'
}}

result = f"Manus deliverable created: {{deliverable_info['uri']}}"
print(f"Created deliverable: {{json.dumps(deliverable_info, indent=2)}}")
"""
        
        sandbox_tool = ImprovedSandboxTool()
        return sandbox_tool._run(creation_code)

# Create tool instances for use in agents
improved_sandbox_exec_tool = ImprovedSandboxTool()
improved_manus_tool = ImprovedManusTool()

