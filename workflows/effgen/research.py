import asyncio
import re
from typing import Optional

from pydantic import BaseModel

from config import settings
from core.logging import logger
from core.storage.jobs import update_job
from core.storage.obsidian import write_note
from core.tracing import observe
from schemas.models import ResearchRequest

_model = None
_model_lock = asyncio.Lock()


class ResearchOutput(BaseModel):
    summary: str
    key_findings: list[str]
    sources: list[str]


async def _get_model():
    global _model
    if _model is not None:
        return _model
    async with _model_lock:
        if _model is not None:
            return _model
        logger.info(f"Loading GGUF model  path={settings.model_path}")
        model = await asyncio.to_thread(_load_model_sync)
        _model = model
        logger.info("Model loaded and cached")
    return _model


def _load_model_sync():
    from effgen import load_model
    return load_model(
        str(settings.model_path),
        engine_config={
            "n_ctx": settings.model_n_ctx,
            "n_gpu_layers": settings.model_n_gpu_layers,
        },
    )


def _run_agent_sync(model, query: str, depth: str) -> tuple[ResearchOutput, list[str], int]:
    from effgen import create_agent

    agent = create_agent("research", model)

    depth_instruction = {
        "quick": "Provide a concise overview with 2-3 key findings.",
        "standard": "Provide a thorough summary with 3-5 key findings.",
        "deep": "Provide a comprehensive analysis with 5-7 key findings and extensive source coverage.",
    }.get(depth, "Provide a thorough summary with 3-5 key findings.")

    task = (
        f"{query} /no_think\n\n"
        f"{depth_instruction} "
        "Structure your response with: a summary paragraph, key findings as a list, "
        "and all sources referenced."
    )

    response = agent.run(task, output_model=ResearchOutput)

    if not response.success:
        raise RuntimeError(
            f"Agent did not succeed after {response.iterations} iterations"
        )

    parsed: Optional[ResearchOutput] = response.metadata.get("parsed")
    if parsed is None:
        parsed = ResearchOutput(
            summary=response.output,
            key_findings=[],
            sources=response.sources or [],
        )

    return parsed, response.sources or [], response.tokens_used


def _slugify(text: str) -> str:
    slug = text.lower()[:60]
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug or "research"


def _format_note(query: str, output: ResearchOutput, sources: list[str]) -> str:
    lines = [f"# {query}", "", "## Summary", "", output.summary, ""]
    if output.key_findings:
        lines += ["## Key Findings", ""]
        lines += [f"- {f}" for f in output.key_findings]
        lines += [""]
    if sources:
        lines += ["## Sources", ""]
        lines += [f"- {s}" for s in sources]
    return "\n".join(lines)


@observe(name="research-workflow")
async def run_research(job_id: str, request: ResearchRequest) -> None:
    logger.info(f"job {job_id} → running  workflow=research  query={request.query!r}")
    await update_job(job_id, "running")

    try:
        model = await _get_model()

        parsed, agent_sources, tokens_used = await asyncio.to_thread(
            _run_agent_sync, model, request.query, request.depth
        )

        # deduplicate, preserve order, prefer parsed sources first
        all_sources = list(dict.fromkeys(parsed.sources + agent_sources))

        slug = _slugify(request.query)
        note_path = write_note(
            "Research",
            slug,
            _format_note(request.query, parsed, all_sources),
            frontmatter={"query": f'"{request.query}"', "depth": request.depth},
        )

        result = {
            "summary": parsed.summary,
            "key_findings": parsed.key_findings,
            "sources": all_sources,
            "tokens_used": tokens_used,
            "obsidian_note": str(note_path),
        }

        await update_job(job_id, "completed", result=result)
        logger.info(
            f"job {job_id} → completed  workflow=research"
            f"  sources={len(all_sources)}  tokens={tokens_used}"
        )

    except Exception as e:
        logger.error(f"job {job_id} → failed  error={e}")
        await update_job(job_id, "failed", error=str(e))
