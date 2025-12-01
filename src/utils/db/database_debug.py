# inspect_db.py
from typing import Optional

from src.utils.db.database import SessionLocal
from src.utils.db import models
from sqlalchemy.orm import Session

class DBInspector:
    """Utility class for inspecting database contents."""

    @staticmethod
    def print_users():
        """Print all users with basic info."""
        db = SessionLocal()
        try:
            users = db.query(models.User).all()
            print("\n=== Users ===")
            for u in users:
                print(f"- id={u.id} | external_id={u.external_id} | created_at={u.created_at}")
        finally:
            db.close()

    @staticmethod
    def print_conversations(user_external_id: Optional[str] = None):
        """Print conversations, optionally filtered by user external_id."""
        db = SessionLocal()
        try:
            q = db.query(models.Conversation).join(models.User)
            if user_external_id:
                q = q.filter(models.User.external_id == user_external_id)

            convs = q.order_by(models.Conversation.created_at).all()
            print("\n=== Conversations ===")
            for c in convs:
                print(
                    f"- id={c.id} | user_external_id={c.user.external_id} | "
                    f"title={c.title} | created_at={c.created_at}"
                )
        finally:
            db.close()

    @staticmethod
    def print_messages(conversation_id: str):
        """Print all messages for a conversation with previews."""
        db = SessionLocal()
        try:
            msgs = (
                db.query(models.Message)
                .filter(models.Message.conversation_id == conversation_id)
                .order_by(models.Message.sequence_no)
                .all()
            )
            print(f"\n=== Messages for conversation {conversation_id} ===")
            for m in msgs:
                preview = (m.content[:80] + "...") if len(m.content) > 80 else m.content
                print(f"{m.sequence_no:03d} | {m.role:<9} | {preview}")
        finally:
            db.close()

    @classmethod
    def inspect_all(cls):
        """Convenience method to print everything."""
        cls.print_users()
        cls.print_conversations()
        print("\n" + "="*50 + "\n")
