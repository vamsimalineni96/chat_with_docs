from typing import List, Optional

from sqlalchemy import func, desc
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.utils.db import models
from src.utils.errors import (
    ConversationServiceError,
    ConversationOwnershipError,
    DatabaseError,
)
from src.utils.services.logger_config import logger


class ConversationService:
    """Service class for managing users, conversations, and messages."""

    @staticmethod
    def get_or_create_user(db: Session, external_id: str) -> models.User:
        """
        Get existing user or create new one by external_id.

        Raises:
            ConversationServiceError: if any DB operation fails.
        """
        try:
            user = (
                db.query(models.User)
                .filter(models.User.external_id == external_id)
                .one_or_none()
            )
            if user:
                logger.info(
                    "User with external_id=%s is present (id=%s)",
                    external_id,
                    user.id,
                )
                return user

            logger.info("Creating the user with external_id=%s", external_id)
            user = models.User(external_id=external_id)
            db.add(user)
            db.commit()
            db.refresh(user)
            return user

        except SQLAlchemyError as e:
            logger.exception(
                "Database error in get_or_create_user(external_id=%s)", external_id
            )
            db.rollback()
            raise ConversationServiceError("Failed to get or create user.") from e

    @staticmethod
    def create_conversation(
        db: Session,
        user: models.User,
        title: Optional[str] = None,
    ) -> models.Conversation:
        """
        Create a new conversation for the user.

        Raises:
            ConversationServiceError: if creation fails.
        """
        try:
            conv = models.Conversation(user_id=user.id, title=title)
            db.add(conv)
            db.commit()
            db.refresh(conv)
            logger.info("Created conversation id=%s for user_id=%s", conv.id, user.id)
            return conv
        except SQLAlchemyError as e:
            logger.exception(
                "Database error in create_conversation(user_id=%s)", user.id
            )
            db.rollback()
            raise ConversationServiceError("Failed to create conversation.") from e

    @staticmethod
    def get_conversation_by_id(
        db: Session,
        conversation_id: str,
        user: models.User,
    ) -> Optional[models.Conversation]:
        """
        Get conversation by ID, filtered by user.

        Returns:
            Conversation or None.

        Raises:
            ConversationServiceError: if the DB query fails.
        """
        try:
            conv = (
                db.query(models.Conversation)
                .filter(
                    models.Conversation.id == conversation_id,
                    models.Conversation.user_id == user.id,
                )
                .one_or_none()
            )
            if conv:
                logger.info(
                    "Fetched conversation id=%s for user_id=%s",
                    conversation_id,
                    user.id,
                )
            else:
                logger.info(
                    "No conversation found with id=%s for user_id=%s",
                    conversation_id,
                    user.id,
                )
            return conv
        except SQLAlchemyError as e:
            logger.exception(
                "Database error in get_conversation_by_id(conv_id=%s, user_id=%s)",
                conversation_id,
                user.id,
            )
            raise ConversationServiceError("Failed to fetch conversation.") from e

    @staticmethod
    def add_message(
        db: Session,
        conversation: models.Conversation,
        user: models.User,
        role: str,
        content: str,
    ) -> models.Message:
        """
        Add a message to conversation with auto-incrementing sequence.

        Raises:
            ConversationOwnershipError: if conversation doesn't belong to user.
            ConversationServiceError: on DB failure.
        """
        if conversation.user_id != user.id:
            logger.error(
                "User %s tried to add message to conversation %s they don't own.",
                user.id,
                conversation.id,
            )
            raise ConversationOwnershipError(
                "Conversation does not belong to this user."
            )

        try:
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

            # bump updated_at
            conversation.updated_at = func.now()

            db.commit()
            db.refresh(msg)

            logger.info(
                "Added message seq_no=%s to conversation_id=%s by user_id=%s",
                next_seq,
                conversation.id,
                user.id,
            )

            return msg
        except SQLAlchemyError as e:
            logger.exception(
                "Database error in add_message(conv_id=%s, user_id=%s)",
                conversation.id,
                user.id,
            )
            db.rollback()
            raise ConversationServiceError("Failed to add message.") from e

    @staticmethod
    def list_conversations(
        db: Session,
        user: models.User,
        limit: int = 100,
    ) -> List[models.Conversation]:
        """
        List a user's conversations, most-recently-updated first.

        Raises:
            ConversationServiceError: on DB failure.
        """
        try:
            convs = (
                db.query(models.Conversation)
                .filter(models.Conversation.user_id == user.id)
                .order_by(desc(models.Conversation.updated_at))
                .limit(limit)
                .all()
            )
            logger.info(
                "Listed %s conversations for user_id=%s", len(convs), user.id
            )
            return convs
        except SQLAlchemyError as e:
            logger.exception(
                "Database error in list_conversations(user_id=%s)", user.id
            )
            raise ConversationServiceError("Failed to list conversations.") from e

    @staticmethod
    def get_recent_messages(
        db: Session,
        conversation: models.Conversation,
        user: models.User,
        limit: int = 20,
    ) -> List[models.Message]:
        """
        Get recent messages in chronological order.

        Raises:
            ConversationOwnershipError: if conversation doesn't belong to user.
            ConversationServiceError: on DB failure.
        """
        if conversation.user_id != user.id:
            logger.error(
                "User %s tried to read messages from conversation %s they don't own.",
                user.id,
                conversation.id,
            )
            raise ConversationOwnershipError(
                "Conversation does not belong to this user."
            )

        try:
            msgs = (
                db.query(models.Message)
                .join(
                    models.Conversation,
                    models.Message.conversation_id == models.Conversation.id,
                )
                .filter(
                    models.Message.conversation_id == conversation.id,
                    models.Conversation.user_id == user.id,
                )
                .order_by(desc(models.Message.sequence_no))
                .limit(limit)
                .all()
            )
            logger.info(
                "Fetched %s recent messages for conversation_id=%s, user_id=%s",
                len(msgs),
                conversation.id,
                user.id,
            )
            return list(reversed(msgs))
        except SQLAlchemyError as e:
            logger.exception(
                "Database error in get_recent_messages(conv_id=%s, user_id=%s)",
                conversation.id,
                user.id,
            )
            raise ConversationServiceError("Failed to fetch recent messages.") from e


_conversation_service_instance = None


def get_conversation_service():
    global _conversation_service_instance
    if _conversation_service_instance is None:
        _conversation_service_instance = ConversationService()
    return _conversation_service_instance
