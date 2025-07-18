"""
AgentFlow Agent Engine - Core agent execution and reasoning system
Based on AgentFlow design specification
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Mock LLM implementation for MVP (replace with real LLM integration)
class MockLLM:
    """Mock LLM for development and testing purposes"""
    
    def __init__(self, provider='openai', model='gpt-3.5-turbo'):
        self.provider = provider
        self.model = model
        self.tokens_used = 0
        self.cost_per_token = 0.0001  # Mock cost
    
    def generate_response(self, prompt: str, context: Dict = None) -> Dict:
        """Generate a mock response based on the prompt"""
        self.tokens_used += len(prompt.split()) * 2  # Mock token calculation
        
        # Simple pattern matching for different types of requests
        if "research" in prompt.lower():
            response = f"Based on my research capabilities, I found relevant information about the topic. Here are the key findings: [Mock research results for the given query]"
        elif "write" in prompt.lower() or "create" in prompt.lower():
            response = f"I have created the requested content based on the provided information and requirements. Here is the result: [Mock generated content]"
        elif "analyze" in prompt.lower():
            response = f"After analyzing the provided data, I can identify several key patterns and insights: [Mock analysis results]"
        elif "summarize" in prompt.lower():
            response = f"Here is a concise summary of the main points: [Mock summary of the content]"
        else:
            response = f"I understand your request and will help you with: {prompt[:100]}... [Mock response based on the request]"
        
        return {
            'response': response,
            'reasoning': [
                "Analyzed the user request and identified the task type",
                "Considered available context and tools",
                "Generated appropriate response based on role and goals",
                "Validated output against success criteria"
            ],
            'tokens_used': self.tokens_used,
            'cost_usd': self.tokens_used * self.cost_per_token
        }


class AgentMemory:
    """Agent memory management system with hierarchical memory levels"""
    
    def __init__(self, config: Dict):
        self.working_memory = []  # Current task context
        self.session_memory = []  # Conversation history
        self.long_term_memory = []  # Persistent knowledge
        self.config = config
        self.max_working_memory = config.get('working_memory_size', 10)
    
    def add_to_working_memory(self, item: Dict):
        """Add item to working memory with size management"""
        self.working_memory.append({
            'timestamp': datetime.utcnow().isoformat(),
            'data': item
        })
        
        # Manage memory size
        if len(self.working_memory) > self.max_working_memory:
            # Move oldest items to session memory if enabled
            if self.config.get('session_memory_enabled', True):
                oldest = self.working_memory.pop(0)
                self.session_memory.append(oldest)
            else:
                self.working_memory.pop(0)
    
    def get_context(self) -> Dict:
        """Get current memory context for agent reasoning"""
        return {
            'working_memory': self.working_memory,
            'session_memory': self.session_memory[-5:] if self.session_memory else [],  # Last 5 items
            'long_term_memory': self.long_term_memory[-3:] if self.long_term_memory else []  # Last 3 items
        }
    
    def clear_working_memory(self):
        """Clear working memory (typically after task completion)"""
        if self.config.get('session_memory_enabled', True):
            self.session_memory.extend(self.working_memory)
        self.working_memory = []


class ToolExecutor:
    """Tool execution system with error handling and validation"""
    
    def __init__(self):
        self.available_tools = {}
        self.execution_history = []
    
    def register_tool(self, tool_name: str, tool_function):
        """Register a tool function"""
        self.available_tools[tool_name] = tool_function
    
    def execute_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Execute a tool with the given parameters"""
        start_time = time.time()
        
        try:
            if tool_name not in self.available_tools:
                return {
                    'success': False,
                    'error': f"Tool '{tool_name}' not found",
                    'execution_time_ms': 0
                }
            
            # Execute the registered tool function
            tool_function = self.available_tools[tool_name]
            result = tool_function(**parameters)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            execution_record = {
                'tool_name': tool_name,
                'parameters': parameters,
                'result': result,
                'execution_time_ms': execution_time,
                'timestamp': datetime.utcnow().isoformat(),
                'success': True
            }
            
            self.execution_history.append(execution_record)
            
            return {
                'success': True,
                'result': result,
                'execution_time_ms': execution_time
            }
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            error_record = {
                'tool_name': tool_name,
                'parameters': parameters,
                'error': str(e),
                'execution_time_ms': execution_time,
                'timestamp': datetime.utcnow().isoformat(),
                'success': False
            }
            
            self.execution_history.append(error_record)
            
            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': execution_time
            }
    
    def _mock_tool_execution(self, tool_name: str, parameters: Dict) -> Any:
        """Mock tool execution for development purposes"""
        if tool_name == 'web_search':
            return {
                'results': [
                    {'title': 'Mock Search Result 1', 'url': 'https://example.com/1', 'snippet': 'Mock search result content'},
                    {'title': 'Mock Search Result 2', 'url': 'https://example.com/2', 'snippet': 'Another mock search result'}
                ],
                'query': parameters.get('query', 'default query')
            }
        elif tool_name == 'text_processor':
            return {
                'processed_text': f"Processed: {parameters.get('text', 'default text')}",
                'word_count': len(parameters.get('text', '').split())
            }
        elif tool_name == 'data_analyzer':
            return {
                'analysis': 'Mock data analysis results',
                'insights': ['Insight 1', 'Insight 2', 'Insight 3'],
                'data_points': parameters.get('data', [])
            }
        else:
            return f"Mock result from {tool_name} with parameters: {parameters}"


class AgentEngine:
    """Core agent execution engine"""
    
    def __init__(self):
        self.llm = MockLLM()
        self.tool_executor = ToolExecutor()
        self.logger = logging.getLogger(__name__)
        
        # Register default tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default system tools"""
        def create_mock_tool(tool_name):
            return lambda **kwargs: f"Mock {tool_name} result"
        
        default_tools = ['web_search', 'text_processor', 'data_analyzer', 'file_reader', 'calculator']
        for tool in default_tools:
            self.tool_executor.register_tool(tool, create_mock_tool(tool))
    
    def execute_agent(self, agent_config: Dict, input_data: Any, context: Dict = None) -> Dict:
        """Execute an agent with the given configuration and input"""
        start_time = time.time()
        
        try:
            # Initialize agent memory
            memory = AgentMemory(agent_config.get('memory_config', {}))
            
            # Add input to working memory
            memory.add_to_working_memory({
                'type': 'input',
                'data': input_data
            })
            
            # Build agent prompt
            prompt = self._build_agent_prompt(agent_config, input_data, memory.get_context())
            
            # Generate response using LLM
            llm_response = self.llm.generate_response(prompt, context)
            
            # Process any tool calls mentioned in the response
            tool_calls = self._extract_tool_calls(llm_response['response'])
            tool_results = []
            
            for tool_call in tool_calls:
                tool_result = self.tool_executor.execute_tool(
                    tool_call['tool_name'], 
                    tool_call['parameters']
                )
                tool_results.append(tool_result)
                
                # Add tool result to memory
                memory.add_to_working_memory({
                    'type': 'tool_result',
                    'tool_name': tool_call['tool_name'],
                    'result': tool_result
                })
            
            # Generate final response if tools were used
            if tool_results:
                final_prompt = self._build_final_response_prompt(
                    agent_config, input_data, llm_response['response'], tool_results
                )
                final_response = self.llm.generate_response(final_prompt)
                output = final_response['response']
                reasoning = llm_response['reasoning'] + final_response['reasoning']
                total_tokens = llm_response['tokens_used'] + final_response['tokens_used']
                total_cost = llm_response['cost_usd'] + final_response['cost_usd']
            else:
                output = llm_response['response']
                reasoning = llm_response['reasoning']
                total_tokens = llm_response['tokens_used']
                total_cost = llm_response['cost_usd']
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return {
                'success': True,
                'output': output,
                'reasoning_log': reasoning,
                'tool_calls': tool_calls,
                'tool_results': tool_results,
                'execution_time_ms': execution_time,
                'tokens_used': total_tokens,
                'cost_usd': total_cost,
                'memory_state': memory.get_context()
            }
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Agent execution failed: {str(e)}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': execution_time,
                'tokens_used': 0,
                'cost_usd': 0.0
            }
    
    def _build_agent_prompt(self, agent_config: Dict, input_data: Any, memory_context: Dict) -> str:
        """Build the prompt for the agent based on its configuration"""
        role = agent_config.get('role', 'Assistant')
        goal = agent_config.get('goal', 'Help the user with their request')
        tools = agent_config.get('tools', [])
        behavioral_params = agent_config.get('behavioral_params', {})
        
        prompt = f"""You are a {role}.

Your goal is: {goal}

Available tools: {', '.join([tool.get('name', str(tool)) for tool in tools]) if tools else 'None'}

Behavioral parameters:
- Creativity level: {behavioral_params.get('creativity_level', 0.7)}
- Risk tolerance: {behavioral_params.get('risk_tolerance', 0.5)}
- Output verbosity: {behavioral_params.get('output_verbosity', 'medium')}
- Reasoning transparency: {behavioral_params.get('reasoning_transparency', True)}

Memory context:
{json.dumps(memory_context, indent=2)}

User request: {input_data}

Please provide a helpful response based on your role and goal. If you need to use any tools, mention them clearly in your response."""
        
        return prompt
    
    def _extract_tool_calls(self, response: str) -> List[Dict]:
        """Extract tool calls from the agent's response (simplified for MVP)"""
        tool_calls = []
        
        # Simple pattern matching for tool calls (in production, use more sophisticated parsing)
        if 'web_search' in response.lower():
            tool_calls.append({
                'tool_name': 'web_search',
                'parameters': {'query': 'extracted search query'}
            })
        
        if 'analyze' in response.lower():
            tool_calls.append({
                'tool_name': 'data_analyzer',
                'parameters': {'data': 'extracted data'}
            })
        
        return tool_calls
    
    def _build_final_response_prompt(self, agent_config: Dict, input_data: Any, 
                                   initial_response: str, tool_results: List[Dict]) -> str:
        """Build prompt for final response after tool execution"""
        return f"""Based on your initial response: "{initial_response}"

And the following tool results:
{json.dumps(tool_results, indent=2)}

Please provide a comprehensive final response to the user's request: {input_data}

Make sure to incorporate the tool results into your response and provide actionable insights."""

