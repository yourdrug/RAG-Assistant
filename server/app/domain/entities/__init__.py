"""Domain entities -- aggregate roots and child entities."""

from domain.entities.conversation import Conversation
from domain.entities.message import Message

__all__ = ["Conversation", "Message"]
