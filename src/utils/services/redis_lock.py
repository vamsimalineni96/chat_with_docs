import time
import uuid
import redis

from src.utils import config


class ConversationLockError(Exception):
    pass


class ConversationLock:
    LOCK_PREFIX = "lock:conversation:"
    DEFAULT_TTL_SECONDS = 60

    def __init__(self, redis_url: str = config.REDIS_URL):
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)

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

        if not wait:
            acquired = self.redis_client.set(key, token, nx=True, ex=ttl_seconds)
            if not acquired:
                raise ConversationLockError("Lock already held for this conversation.")
            return token

        # Wait mode: retry until timeout
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            acquired = self.redis_client.set(key, token, nx=True, ex=ttl_seconds)
            if acquired:
                return token
            time.sleep(wait_interval)

        raise ConversationLockError("Timeout while waiting for conversation lock.")

    def release_conversation_lock(self, conversation_id: str, token: str) -> None:
        """
        Release the lock only if the token matches (to avoid releasing someone else's lock).
        """
        key = self._lock_key(conversation_id)
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
