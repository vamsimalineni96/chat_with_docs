from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from src.utils import config
from src.utils.services.logger_config import logger

# For simple setup we'll use sync SQLAlchemy
engine = create_engine(
    config.DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    """
    FastAPI dependency:
    from app.db.database import get_db
    """
    logger.info("Accessing the postgres database")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
