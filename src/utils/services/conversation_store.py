from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from src.utils.db import models


class ConversationService:
    """Service class for managing users, conversations, and messages."""

    @staticmethod
    def get_or_create_user(db: Session, external_id: str) -> models.User:
        """Get existing user or create new one by external_id."""
        user = (
            db.query(models.User)
            .filter(models.User.external_id == external_id)
            .one_or_none()
        )
        if user:
            return user
        
        user = models.User(external_id=external_id)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def create_conversation(
        db: Session,
        user: models.User,
        title: Optional[str] = None,
    ) -> models.Conversation:
        """Create a new conversation for the user."""
        conv = models.Conversation(user_id=user.id, title=title)
        db.add(conv)
        db.commit()
        db.refresh(conv)
        return conv

    @staticmethod
    def get_conversation_by_id(
        db: Session,
        conversation_id: str,
        user: Optional[models.User] = None,
    ) -> Optional[models.Conversation]:
        """Get conversation by ID, optionally filtered by user."""
        q = db.query(models.Conversation).filter(models.Conversation.id == conversation_id)
        if user is not None:
            q = q.filter(models.Conversation.user_id == user.id)
        return q.one_or_none()

    @staticmethod
    def add_message(
        db: Session,
        conversation: models.Conversation,
        role: str,
        content: str,
    ) -> models.Message:
        """Add a message to conversation with auto-incrementing sequence."""
        # Find next sequence_no
        last_seq = (
            db.query(func.max(models.Message.sequence_no))
            .filter(models.Message.conversation_id == conversation.id)
            .scalar()
        )
        next_seq = 1 if last_seq is None else last_seq + 1

        msg = models.Message(
            conversation_id=conversation.id,
            role=role,
            content=content,
            sequence_no=next_seq,
        )
        db.add(msg)

        # Bump updated_at on conversation
        conversation.updated_at = func.now()

        db.commit()
        db.refresh(msg)
        return msg

    @staticmethod
    def get_recent_messages(
        db: Session,
        conversation: models.Conversation,
        limit: int = 20,
    ) -> List[models.Message]:
        """Get recent messages in chronological order."""
        msgs = (
            db.query(models.Message)
            .filter(models.Message.conversation_id == conversation.id)
            .order_by(desc(models.Message.sequence_no))
            .limit(limit)
            .all()
        )
        # Reverse to chronological order
        return list(reversed(msgs))
