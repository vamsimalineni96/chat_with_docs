import time
import uuid

import redis

from src.utils import config
from src.utils.errors import RedisLockError, ConversationLockError
from src.utils.services.logger_config import logger




class ConversationLock:
    LOCK_PREFIX = "lock:conversation:"
    DEFAULT_TTL_SECONDS = 60

    def __init__(self, redis_url: str = config.REDIS_URL):
        try:
            self.redis_client = redis.Redis.from_url(
                redis_url, decode_responses=True
            )
        except Exception as e:
            logger.exception("Failed to connect to Redis: %s", e)
            raise RedisLockError("Failed to connect to Redis for conversation locking.") from e

    def _lock_key(self, conversation_id: str) -> str:
        return f"{self.LOCK_PREFIX}{conversation_id}"

    def acquire_conversation_lock(
        self,
        conversation_id: str,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        wait: bool = False,
        wait_timeout: float = 5.0,
        wait_interval: float = 0.1,
    ) -> str:
        """
        Acquire a lock for the given conversation_id.

        Returns a lock_token (string) to be used to release the lock.
        If wait=False and lock is already held, raises ConversationLockError.
        """
        key = self._lock_key(conversation_id)
        token = str(uuid.uuid4())

        try:
            if not wait:
                acquired = self.redis_client.set(key, token, nx=True, ex=ttl_seconds)
                if not acquired:
                    raise ConversationLockError(
                        "Lock already held for this conversation."
                    )
                return token

            deadline = time.time() + wait_timeout
            while time.time() < deadline:
                acquired = self.redis_client.set(key, token, nx=True, ex=ttl_seconds)
                if acquired:
                    return token
                time.sleep(wait_interval)

            raise ConversationLockError(
                "Timeout while waiting for conversation lock."
            )
        except ConversationLockError:
            raise
        except redis.RedisError as e:
            logger.exception("Redis error while acquiring conversation lock: %s", e)
            raise ConversationLockError("Redis error while acquiring lock.") from e
        except Exception as e:
            logger.exception("Unexpected error while acquiring conversation lock: %s", e)
            raise ConversationLockError("Unexpected error while acquiring lock.") from e

    def release_conversation_lock(self, conversation_id: str, token: str) -> None:
        """
        Release the lock only if the token matches (to avoid releasing someone else's lock).
        """
        key = self._lock_key(conversation_id)
        try:
            with self.redis_client.pipeline() as pipe:
                while True:
                    try:
                        pipe.watch(key)
                        current_token = pipe.get(key)
                        if current_token is None:
                            pipe.unwatch()
                            return
                        if current_token != token:
                            pipe.unwatch()
                            raise ConversationLockError("Lock token mismatch.")

                        pipe.multi()
                        pipe.delete(key)
                        pipe.execute()
                        return
                    except redis.WatchError:
                        # value changed between WATCH and EXEC, retry
                        continue
        except ConversationLockError:
            raise
        except redis.RedisError as e:
            logger.exception("Redis error while releasing conversation lock: %s", e)
            raise ConversationLockError("Redis error while releasing lock.") from e
        except Exception as e:
            logger.exception("Unexpected error while releasing conversation lock: %s", e)
            raise ConversationLockError("Unexpected error while releasing lock.") from e


_redis_store_instance = None


def get_redis_lock():
    global _redis_store_instance
    if _redis_store_instance is None:
        _redis_store_instance = ConversationLock()
    return _redis_store_instance
