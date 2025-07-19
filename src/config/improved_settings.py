"""
Improved configuration settings for AgentFlow backend.
Integrates with existing Flask configuration while adding new AI model settings.
"""
import os
from pydantic_settings import BaseSettings

class ImprovedSettings(BaseSettings):
    """Enhanced runtime configuration for AgentFlow with AI model support."""
    
    # Existing Flask settings (maintain compatibility)
    flask_env: str = os.getenv('FLASK_ENV', 'development')
    secret_key: str = os.getenv('SECRET_KEY', 'dev-secret-key')
    database_url: str = os.getenv('DATABASE_URL', 'sqlite:///agentflow.db')
    
    # AI Model API Keys
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    xai_api_key: str = os.getenv('XAI_API_KEY', '')  # For Grok-4
    e2b_api_key: str | None = os.getenv('E2B_API_KEY', None)
    
    # AI Model Configuration
    model_openai: str = os.getenv('MODEL_OPENAI', 'gpt-4o')
    model_grok: str = os.getenv('MODEL_GROK', 'grok-4')
    
    # Sandbox Configuration
    use_restrictedpython_fallback: bool = os.getenv('USE_RESTRICTEDPYTHON_FALLBACK', 'true').lower() == 'true'
    
    # CrewAI Configuration
    crew_max_iterations: int = int(os.getenv('CREW_MAX_ITERATIONS', '10'))
    crew_verbose: bool = os.getenv('CREW_VERBOSE', 'false').lower() == 'true'

    class Config:
        env_file = ".env"

# Global settings instance
improved_settings = ImprovedSettings()

