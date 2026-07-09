import asyncio
import threading
from typing import AsyncIterator

from config import settings
from core.logging import logger
from schemas.models import RagChatRequest

_RAG_SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. When answering, ALWAYS consult the "
    "knowledge base first using the retrieval tool. Cite sources inline using [1], [2], "
    "... markers matching the returned citation list. If the knowledge base does not "
    "contain the answer, say so explicitly rather than guessing."
)

_FA_PREFIX = "Final Answer:"

_model = None
_model_lock = asyncio.Lock()


def _load_model():
    from effgen import load_model
    return load_model(
        settings.llm_model,
        provider="openai",
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


async def _get_model():
    global _model
    if _model is not None:
        return _model
    async with _model_lock:
        if _model is not None:
            return _model
        logger.info(f"Connecting to LLM  base_url={settings.llm_base_url}  model={settings.llm_model}")
        model = await asyncio.to_thread(_load_model)
        _model = model
        logger.info("LLM connection ready")
    return _model


def _run_stream(
    model,
    index_path: str,
    message: str,
    history: list,
    loop: asyncio.AbstractEventLoop,
    q: "asyncio.Queue[dict | None]",
) -> None:
    def emit(event: dict) -> None:
        asyncio.run_coroutine_threadsafe(q.put(event), loop)

    try:
        from effgen import create_agent
        from effgen.tools.builtin import Retrieval

        retrieval = Retrieval(index_path=index_path)
        agent = create_agent(
            "minimal",
            model,
            extra_tools=[retrieval],
            system_prompt=_RAG_SYSTEM_PROMPT,
            max_context_length=settings.llm_max_tokens,
        )

        for turn in history:
            if turn.role == "user":
                agent.short_term_memory.add_user_message(turn.content)
            else:
                agent.short_term_memory.add_assistant_message(turn.content)

        run_kwargs: dict = {}
        if settings.llm_max_tokens is not None:
            run_kwargs["max_tokens"] = settings.llm_max_tokens

        buf = ""
        in_answer = False

        for chunk in agent.stream(message, **run_kwargs):
            if in_answer:
                if chunk:
                    emit({"type": "answer_chunk", "text": chunk})
                continue

            buf += chunk
            if _FA_PREFIX in buf:
                in_answer = True
                idx = buf.index(_FA_PREFIX)
                tail = buf[idx + len(_FA_PREFIX):].lstrip()
                if tail:
                    emit({"type": "answer_chunk", "text": tail})
                buf = ""
            else:
                safe = len(buf) - (len(_FA_PREFIX) - 1)
                if safe > 0:
                    buf = buf[safe:]

        emit({"type": "done"})

    except Exception as e:
        logger.error(f"RAG chat stream error: {e}")
        emit({"type": "error", "text": str(e)})
    finally:
        asyncio.run_coroutine_threadsafe(q.put(None), loop)


async def stream_rag_chat(request: RagChatRequest) -> AsyncIterator[str]:
    """Async SSE generator — yields plain-text data lines ending with [DONE]."""
    model = await _get_model()
    index_path = str(settings.brain_vault_path / "rag" / request.collection_id / "index.pkl")

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    thread = threading.Thread(
        target=_run_stream,
        args=(model, index_path, request.message, request.history, loop, q),
        daemon=True,
    )
    thread.start()

    while True:
        event = await q.get()
        if event is None:
            yield "data: [DONE]\n\n"
            return
        event_type = event.get("type")
        if event_type == "answer_chunk":
            yield f"data: {event['text']}\n\n"
        elif event_type == "done":
            yield "data: [DONE]\n\n"
            return
        elif event_type == "error":
            yield f"data: [ERROR] {event['text']}\n\n"
            yield "data: [DONE]\n\n"
            return
        # thinking_chunk, tool_call, observation — omit from SSE output
