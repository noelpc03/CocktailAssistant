"""
Agent interface defining the basic structure for all agents in the system.
"""
import abc
from typing import Dict, Any, List, Optional


class Agent(abc.ABC):
    """Base abstract class for all agents"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        """
        Initialize an agent with an ID and configuration
        
        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dictionary for the agent
        """
        self.agent_id = agent_id
        self.config = config or {}
        self.inbox = []
        self._running = False
        
    @abc.abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message and return a response.
        
        Args:
            message: The message to process
            
        Returns:
            The response message
        """
        pass
    
    async def send_message(self, target_agent: str, content: Dict[str, Any]) -> None:
        """
        Send a message to another agent through the message broker
        
        Args:
            target_agent: The ID of the agent to send the message to
            content: The content of the message
        """
        from agents.common.message_broker import MessageBroker
        import logging
        logger = logging.getLogger(__name__)
        
        action = content.get("action", "unknown")
        operation_id = content.get("operation_id", "none")
        
        logger.info(f"Agent {self.agent_id} sending message to {target_agent}: action={action}, operation_id={operation_id}")
        
        message = {
            "sender": self.agent_id,
            "recipient": target_agent,
            "content": content
        }
        
        await MessageBroker().publish_message(message)
    
    async def receive_message(self, message: Dict[str, Any]) -> None:
        """
        Receive a message from another agent
        
        Args:
            message: The message received
        """
        import logging
        logger = logging.getLogger(__name__)
        
        sender = message.get("sender", "unknown")
        content = message.get("content", {})
        action = content.get("action", "unknown")
        
        logger.info(f"Agent {self.agent_id} received message from {sender}: action={action}")
        
        self.inbox.append(message)
        response = await self.process_message(message)
        
        if response and "recipient" in response:
            resp_content = response.get("content", {})
            resp_action = resp_content.get("action", "unknown")
            logger.info(f"Agent {self.agent_id} sending response to {response['recipient']}: action={resp_action}")
            await self.send_message(response["recipient"], response["content"])
    
    async def start(self) -> None:
        """Start the agent's processing loop"""
        self._running = True
        # Agent-specific startup code should be implemented in subclasses
    
    async def stop(self) -> None:
        """Stop the agent's processing loop"""
        self._running = False
        # Agent-specific shutdown code should be implemented in subclasses
