import asyncio
import json
import re
from typing import Optional

from pydantic import BaseModel

from config import settings
from core.logging import logger
from core.storage.jobs import update_job
from core.tracing import observe
from schemas.models import LearningPlanRequest


class _ActivityCard(BaseModel):
    day: str
    title: str
    description: str
    durationMinutes: int
    taxonomyNodes: list[str]
    neurodivergentTips: str
    materials: list[str]


class _LearningPlan(BaseModel):
    activities: list[_ActivityCard]


def _load_model():
    from effgen import load_model
    return load_model(
        settings.llm_model,
        provider="openai",
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


_model = None
_model_lock = asyncio.Lock()


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


def _build_prompt(request: LearningPlanRequest) -> str:
    child = request.child
    lines = [
        f"Create a 5-day weekly learning plan for {child.name}.",
        f"Child notes: {child.notes or 'none'}",
        f"Week starting: {request.week_start}",
    ]
    if request.focus_note:
        lines.append(f"Parent's focus note: {request.focus_note}")

    if request.gap_nodes:
        lines.append("\nConcepts to reinforce this week (use their IDs in taxonomyNodes):")
        for node in request.gap_nodes[:12]:
            name = node.name or node.id
            lines.append(f"  - {node.id} — {name}: {node.description}")

    if request.recent_logs:
        lines.append("\nRecent activities (avoid repeating these):")
        for log in request.recent_logs[:5]:
            lines.append(f"  - {log.date}: {log.description}")

    lines += [
        "",
        "Rules:",
        "- Exactly 5 activities, one each for Monday through Friday (day field = full weekday name).",
        "- Age-appropriate for Pre-K / Kindergarten (ages 4-6), hands-on and playful.",
        "- durationMinutes: integer between 20 and 45.",
        "- taxonomyNodes: list of 1–3 node IDs from the concept list above (the mt_xxx strings).",
        "- materials: simple household items; empty list if none needed.",
        "- neurodivergentTips: one concrete tip for sensory, attention, or pacing needs.",
        "- Vary the activities across the week; do not repeat the same format two days in a row.",
    ]
    return "\n".join(lines)


def _run_agent_sync(model, request: LearningPlanRequest) -> _LearningPlan:
    from effgen import create_agent

    agent = create_agent("minimal", model, extra_tools=[], tool_calling_mode="react", max_iterations=5)
    prompt = _build_prompt(request)
    response = agent.run(prompt, output_model=_LearningPlan)

    if not response.success:
        reason = (response.metadata or {}).get("reason", "unknown")
        raise RuntimeError(
            f"Agent did not succeed after {response.iterations} iterations  reason={reason}"
        )

    parsed: Optional[_LearningPlan] = (response.metadata or {}).get("parsed")
    if parsed is not None:
        return parsed

    # Fallback: the model likely emitted a raw JSON array instead of {"activities": [...]}.
    # Regex-extract the first array from the raw output and coerce it.
    output = response.output or ""
    array_match = re.search(r'\[[\s\S]*\]', output)
    if array_match:
        try:
            items = json.loads(array_match.group())
            if isinstance(items, list) and items:
                return _LearningPlan(activities=[_ActivityCard(**a) for a in items])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Also try an object wrapper with an "activities" key
    obj_match = re.search(r'\{[\s\S]*\}', output)
    if obj_match:
        try:
            data = json.loads(obj_match.group())
            if isinstance(data, dict) and "activities" in data:
                return _LearningPlan(activities=[_ActivityCard(**a) for a in data["activities"]])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    raise RuntimeError(
        f"No structured output from learning-plan agent (output: {output[:300]!r})"
    )


@observe(name="learning-plan-workflow")
async def run_learning_plan(job_id: str, request: LearningPlanRequest) -> None:
    child_name = request.child.name
    logger.info(f"job {job_id} → running  workflow=learning-plan  child={child_name!r}")
    await update_job(job_id, "running")

    try:
        model = await _get_model()

        try:
            plan = await asyncio.to_thread(_run_agent_sync, model, request)
        except Exception as e:
            if "connection" in str(e).lower():
                global _model
                _model = None
                logger.warning("LLM connection lost — cleared model cache")
            raise

        # Store as a JSON string so kids-learning's parsePlanResult can regex-extract the array.
        # update_job JSON-encodes whatever we pass, so passing a string double-encodes it —
        # get_job decodes it back to a string, which is exactly what the TS client expects.
        result_str = json.dumps([a.model_dump() for a in plan.activities])

        await update_job(job_id, "completed", result=result_str)  # type: ignore[arg-type]
        logger.info(
            f"job {job_id} → completed  workflow=learning-plan"
            f"  child={child_name!r}  activities={len(plan.activities)}"
        )

    except Exception as e:
        logger.error(f"job {job_id} → failed  error={e}")
        await update_job(job_id, "failed", error=str(e))
