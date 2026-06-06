from config import settings
from core.logging import logger

_tracing_enabled = False


def setup_tracing() -> bool:
    global _tracing_enabled
    if not settings.langfuse_secret_key or not settings.langfuse_public_key:
        logger.warning("LangFuse keys not set — tracing disabled")
        return False

    try:
        import langfuse
        langfuse.Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host,
        )
        _tracing_enabled = True
        logger.info(f"LangFuse tracing enabled  host={settings.langfuse_host}")
        return True
    except Exception as e:
        logger.error(f"LangFuse init failed: {e}")
        return False


def is_tracing_enabled() -> bool:
    return _tracing_enabled


def observe(*args, **kwargs):
    """Conditional @observe — no-ops when tracing is disabled."""
    if _tracing_enabled:
        from langfuse.decorators import observe as _observe
        return _observe(*args, **kwargs)

    def passthrough(fn):
        return fn

    if args and callable(args[0]):
        return args[0]
    return passthrough
