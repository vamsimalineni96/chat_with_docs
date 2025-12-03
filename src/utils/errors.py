# errors.py
class InferenceError(Exception):
    """Raised when the LLM/NIM inference fails in a way the client should know about."""
    pass
