"""
Agent Messenger - Inter-Agent Communication System
Handles communication, context sharing, and coordination between agents
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class MessageType(Enum):
    """Types of inter-agent messages"""
    TASK_DELEGATION = "task_delegation"
    RESULT_SHARING = "result_sharing"
    FEEDBACK_REQUEST = "feedback_request"
    FEEDBACK_RESPONSE = "feedback_response"
    COORDINATION = "coordination"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AgentMessage:
    """Inter-agent message structure"""
    id: str
    sender_agent: str
    recipient_agent: str
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    timestamp: float
    requires_response: bool = False
    response_timeout: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

@dataclass
class MessageResponse:
    """Response to an agent message"""
    message_id: str
    responder_agent: str
    response_content: Dict[str, Any]
    timestamp: float
    success: bool

class AgentMessenger:
    """
    Central messaging system for inter-agent communication
    """
    
    def __init__(self):
        self.message_queue: List[AgentMessage] = []
        self.message_history: List[AgentMessage] = []
        self.responses: Dict[str, MessageResponse] = {}
        self.agent_subscriptions: Dict[str, List[MessageType]] = {}
        self.communication_stats = {
            'total_messages': 0,
            'successful_communications': 0,
            'failed_communications': 0,
            'average_response_time': 0.0
        }
    
    def send_message(self, 
                    sender: str, 
                    recipient: str, 
                    message_type: MessageType, 
                    content: Dict[str, Any],
                    priority: MessagePriority = MessagePriority.MEDIUM,
                    requires_response: bool = False,
                    response_timeout: float = 30.0,
                    context: Dict[str, Any] = None) -> str:
        """
        Send a message from one agent to another
        """
        message_id = f"msg_{int(time.time() * 1000)}_{len(self.message_queue)}"
        
        message = AgentMessage(
            id=message_id,
            sender_agent=sender,
            recipient_agent=recipient,
            message_type=message_type,
            priority=priority,
            content=content,
            timestamp=time.time(),
            requires_response=requires_response,
            response_timeout=response_timeout,
            context=context
        )
        
        # Add to queue and history
        self.message_queue.append(message)
        self.message_history.append(message)
        
        # Update stats
        self.communication_stats['total_messages'] += 1
        
        # Log the communication
        self._log_communication(message)
        
        return message_id
    
    def get_messages_for_agent(self, agent_name: str, message_types: List[MessageType] = None) -> List[AgentMessage]:
        """
        Get pending messages for a specific agent
        """
        messages = []
        
        for message in self.message_queue:
            if message.recipient_agent == agent_name:
                if message_types is None or message.message_type in message_types:
                    messages.append(message)
        
        # Sort by priority and timestamp
        messages.sort(key=lambda m: (m.priority.value, m.timestamp))
        
        return messages
    
    def send_response(self, 
                     message_id: str, 
                     responder: str, 
                     response_content: Dict[str, Any],
                     success: bool = True) -> bool:
        """
        Send a response to a message
        """
        # Find the original message
        original_message = None
        for message in self.message_history:
            if message.id == message_id:
                original_message = message
                break
        
        if not original_message:
            return False
        
        if not original_message.requires_response:
            return False
        
        # Create response
        response = MessageResponse(
            message_id=message_id,
            responder_agent=responder,
            response_content=response_content,
            timestamp=time.time(),
            success=success
        )
        
        self.responses[message_id] = response
        
        # Update communication stats
        if success:
            self.communication_stats['successful_communications'] += 1
        else:
            self.communication_stats['failed_communications'] += 1
        
        # Calculate response time
        response_time = response.timestamp - original_message.timestamp
        self._update_average_response_time(response_time)
        
        return True
    
    def get_response(self, message_id: str, timeout: float = 30.0) -> Optional[MessageResponse]:
        """
        Get response to a message (with timeout)
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if message_id in self.responses:
                return self.responses[message_id]
            time.sleep(0.1)  # Small delay to prevent busy waiting
        
        return None
    
    def broadcast_message(self, 
                         sender: str, 
                         message_type: MessageType, 
                         content: Dict[str, Any],
                         recipients: List[str] = None,
                         priority: MessagePriority = MessagePriority.MEDIUM) -> List[str]:
        """
        Broadcast a message to multiple agents
        """
        message_ids = []
        
        # If no specific recipients, broadcast to all subscribed agents
        if recipients is None:
            recipients = self._get_subscribed_agents(message_type)
        
        for recipient in recipients:
            if recipient != sender:  # Don't send to self
                message_id = self.send_message(
                    sender=sender,
                    recipient=recipient,
                    message_type=message_type,
                    content=content,
                    priority=priority
                )
                message_ids.append(message_id)
        
        return message_ids
    
    def subscribe_agent(self, agent_name: str, message_types: List[MessageType]):
        """
        Subscribe an agent to specific message types
        """
        if agent_name not in self.agent_subscriptions:
            self.agent_subscriptions[agent_name] = []
        
        for message_type in message_types:
            if message_type not in self.agent_subscriptions[agent_name]:
                self.agent_subscriptions[agent_name].append(message_type)
    
    def unsubscribe_agent(self, agent_name: str, message_types: List[MessageType] = None):
        """
        Unsubscribe an agent from message types
        """
        if agent_name not in self.agent_subscriptions:
            return
        
        if message_types is None:
            # Unsubscribe from all
            self.agent_subscriptions[agent_name] = []
        else:
            for message_type in message_types:
                if message_type in self.agent_subscriptions[agent_name]:
                    self.agent_subscriptions[agent_name].remove(message_type)
    
    def mark_message_processed(self, message_id: str):
        """
        Mark a message as processed and remove from queue
        """
        self.message_queue = [msg for msg in self.message_queue if msg.id != message_id]
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """
        Get communication statistics
        """
        return {
            **self.communication_stats,
            'pending_messages': len(self.message_queue),
            'total_responses': len(self.responses),
            'active_agents': len(self.agent_subscriptions)
        }
    
    def get_agent_communication_history(self, agent_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get communication history for a specific agent
        """
        history = []
        
        for message in self.message_history[-limit:]:
            if message.sender_agent == agent_name or message.recipient_agent == agent_name:
                history.append({
                    'message_id': message.id,
                    'sender': message.sender_agent,
                    'recipient': message.recipient_agent,
                    'type': message.message_type.value,
                    'priority': message.priority.value,
                    'timestamp': message.timestamp,
                    'content_summary': str(message.content)[:100],
                    'requires_response': message.requires_response,
                    'has_response': message.id in self.responses
                })
        
        return history
    
    def create_coordination_context(self, execution_id: str, participating_agents: List[str]) -> Dict[str, Any]:
        """
        Create a coordination context for multi-agent collaboration
        """
        context = {
            'execution_id': execution_id,
            'participating_agents': participating_agents,
            'coordination_started': time.time(),
            'shared_data': {},
            'agent_status': {agent: 'ready' for agent in participating_agents},
            'communication_log': []
        }
        
        # Subscribe all agents to coordination messages
        for agent in participating_agents:
            self.subscribe_agent(agent, [MessageType.COORDINATION, MessageType.STATUS_UPDATE])
        
        return context
    
    def update_coordination_context(self, 
                                  execution_id: str, 
                                  agent_name: str, 
                                  status: str, 
                                  data: Dict[str, Any] = None):
        """
        Update coordination context with agent status and data
        """
        # This would typically be stored in a shared context store
        # For now, we'll broadcast the update
        content = {
            'execution_id': execution_id,
            'agent_name': agent_name,
            'status': status,
            'data': data or {},
            'timestamp': time.time()
        }
        
        # Broadcast status update to all participating agents
        self.broadcast_message(
            sender=agent_name,
            message_type=MessageType.STATUS_UPDATE,
            content=content,
            priority=MessagePriority.HIGH
        )
    
    def request_feedback(self, 
                        requester: str, 
                        target_agent: str, 
                        work_to_review: Dict[str, Any],
                        feedback_criteria: List[str] = None) -> str:
        """
        Request feedback from another agent
        """
        content = {
            'work_to_review': work_to_review,
            'feedback_criteria': feedback_criteria or [
                'quality', 'completeness', 'accuracy', 'relevance'
            ],
            'requester_context': {
                'agent_type': requester,
                'request_timestamp': time.time()
            }
        }
        
        message_id = self.send_message(
            sender=requester,
            recipient=target_agent,
            message_type=MessageType.FEEDBACK_REQUEST,
            content=content,
            priority=MessagePriority.HIGH,
            requires_response=True,
            response_timeout=60.0
        )
        
        return message_id
    
    def provide_feedback(self, 
                        message_id: str, 
                        reviewer: str, 
                        feedback: Dict[str, Any]) -> bool:
        """
        Provide feedback in response to a feedback request
        """
        feedback_content = {
            'feedback': feedback,
            'reviewer': reviewer,
            'review_timestamp': time.time(),
            'feedback_summary': self._generate_feedback_summary(feedback)
        }
        
        return self.send_response(
            message_id=message_id,
            responder=reviewer,
            response_content=feedback_content,
            success=True
        )
    
    def _get_subscribed_agents(self, message_type: MessageType) -> List[str]:
        """
        Get list of agents subscribed to a message type
        """
        subscribed = []
        
        for agent, subscriptions in self.agent_subscriptions.items():
            if message_type in subscriptions:
                subscribed.append(agent)
        
        return subscribed
    
    def _update_average_response_time(self, response_time: float):
        """
        Update average response time statistic
        """
        current_avg = self.communication_stats['average_response_time']
        total_responses = len(self.responses)
        
        if total_responses == 1:
            self.communication_stats['average_response_time'] = response_time
        else:
            # Calculate running average
            new_avg = ((current_avg * (total_responses - 1)) + response_time) / total_responses
            self.communication_stats['average_response_time'] = new_avg
    
    def _generate_feedback_summary(self, feedback: Dict[str, Any]) -> str:
        """
        Generate a summary of feedback content
        """
        if 'overall_score' in feedback:
            score = feedback['overall_score']
            if score >= 8:
                return f"Excellent work (Score: {score}/10)"
            elif score >= 6:
                return f"Good work with room for improvement (Score: {score}/10)"
            else:
                return f"Needs significant improvement (Score: {score}/10)"
        
        return "Feedback provided"
    
    def _log_communication(self, message: AgentMessage):
        """
        Log communication for debugging and analysis
        """
        log_entry = {
            'timestamp': message.timestamp,
            'sender': message.sender_agent,
            'recipient': message.recipient_agent,
            'type': message.message_type.value,
            'priority': message.priority.value,
            'requires_response': message.requires_response
        }
        
        # In production, this would go to a proper logging system
        print(f"📨 Agent Communication: {message.sender_agent} → {message.recipient_agent} ({message.message_type.value})")

# Global messenger instance
agent_messenger = AgentMessenger()

# Helper functions for common communication patterns
def delegate_task(manager_agent: str, target_agent: str, task_description: str, context: Dict[str, Any] = None) -> str:
    """Helper function to delegate a task to another agent"""
    content = {
        'task_description': task_description,
        'delegation_context': context or {},
        'expected_deliverables': context.get('expected_deliverables', []) if context else []
    }
    
    return agent_messenger.send_message(
        sender=manager_agent,
        recipient=target_agent,
        message_type=MessageType.TASK_DELEGATION,
        content=content,
        priority=MessagePriority.HIGH,
        requires_response=True
    )

def share_results(sender_agent: str, recipient_agent: str, results: Dict[str, Any], context: Dict[str, Any] = None) -> str:
    """Helper function to share results between agents"""
    content = {
        'results': results,
        'result_context': context or {},
        'sender_agent_type': sender_agent
    }
    
    return agent_messenger.send_message(
        sender=sender_agent,
        recipient=recipient_agent,
        message_type=MessageType.RESULT_SHARING,
        content=content,
        priority=MessagePriority.MEDIUM
    )

def coordinate_agents(coordinator: str, agents: List[str], coordination_plan: Dict[str, Any]) -> str:
    """Helper function to coordinate multiple agents"""
    content = {
        'coordination_plan': coordination_plan,
        'participating_agents': agents,
        'coordinator': coordinator
    }
    
    message_ids = agent_messenger.broadcast_message(
        sender=coordinator,
        message_type=MessageType.COORDINATION,
        content=content,
        recipients=agents,
        priority=MessagePriority.HIGH
    )
    
    return message_ids[0] if message_ids else None

