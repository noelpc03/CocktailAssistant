"""
Message broker for inter-agent communication.
"""
import asyncio
from typing import Dict, Any, List, Callable, Optional
import logging

logger = logging.getLogger(__name__)

class MessageBroker:
    """
    Message broker singleton for handling inter-agent communication.
    Implements a publisher-subscriber pattern.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MessageBroker, cls).__new__(cls)
            cls._instance._subscribers = {}
            cls._instance._message_queue = asyncio.Queue()
            cls._instance._running = False
            cls._instance._task = None
        return cls._instance
    
    async def publish_message(self, message: Dict[str, Any]) -> None:
        """
        Publish a message to the broker's queue
        
        Args:
            message: The message to publish
        """
        if not isinstance(message, dict):
            logger.error(f"Invalid message format: {message}")
            return
            
        if "recipient" not in message:
            logger.error(f"Message missing recipient: {message}")
            return
            
        await self._message_queue.put(message)
        logger.debug(f"Published message: {message['sender']} -> {message['recipient']}")
    
    def subscribe(self, agent_id: str, callback: Callable) -> None:
        """
        Subscribe an agent to receive messages
        
        Args:
            agent_id: The agent ID to subscribe
            callback: The callback function to invoke when a message is received
        """
        self._subscribers[agent_id] = callback
        logger.info(f"Agent {agent_id} subscribed to message broker")
    
    def unsubscribe(self, agent_id: str) -> None:
        """
        Unsubscribe an agent from receiving messages
        
        Args:
            agent_id: The agent ID to unsubscribe
        """
        if agent_id in self._subscribers:
            del self._subscribers[agent_id]
            logger.info(f"Agent {agent_id} unsubscribed from message broker")
    
    async def _process_message_queue(self) -> None:
        """Process messages in the queue and dispatch them to subscribers"""
        while self._running:
            try:
                message = await self._message_queue.get()
                recipient = message["recipient"]
                
                if recipient == "broadcast":
                    # Send to all subscribers
                    for agent_id, callback in self._subscribers.items():
                        if agent_id != message["sender"]:
                            await callback(message)
                elif recipient in self._subscribers:
                    # Send to specific recipient
                    await self._subscribers[recipient](message)
                else:
                    logger.warning(f"No subscriber found for recipient: {recipient}")
                
                self._message_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def start(self) -> None:
        """Start the message broker"""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._process_message_queue())
            logger.info("Message broker started")
    
    async def stop(self) -> None:
        """Stop the message broker"""
        if self._running:
            self._running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None
            logger.info("Message broker stopped")
