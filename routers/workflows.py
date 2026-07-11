from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core.auth import get_current_user
from core.logging import logger
from core.storage.jobs import create_job, get_job, list_jobs
from schemas.models import (
    CodingRequest,
    JobCreatedResponse,
    JobStatusResponse,
    LearningPlanRequest,
    RagChatRequest,
    RagIngestRequest,
    ResearchRequest,
    SoccerAnalystRequest,
    SoccerAnalystResponse,
)

router = APIRouter(tags=["workflows"])


@router.post("/research", response_model=JobCreatedResponse)
async def trigger_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
) -> JobCreatedResponse:
    job_id = await create_job("research", request.model_dump(), user_id=current_user["id"])
    from workflows.effgen.research import run_research
    background_tasks.add_task(run_research, job_id, request)
    logger.info(f"job {job_id} queued  workflow=research  query={request.query!r}")
    return JobCreatedResponse(job_id=job_id)


@router.post("/coding", response_model=JobCreatedResponse)
async def trigger_coding(
    request: CodingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
) -> JobCreatedResponse:
    job_id = await create_job("coding", request.model_dump(), user_id=current_user["id"])
    from workflows.coding.app_builder import run_app_builder
    background_tasks.add_task(run_app_builder, job_id, request)
    logger.info(f"job {job_id} queued  workflow=coding  idea={request.idea[:60]!r}")
    return JobCreatedResponse(job_id=job_id)


@router.post("/briefing", response_model=JobCreatedResponse)
async def trigger_briefing(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
) -> JobCreatedResponse:
    job_id = await create_job("briefing", {}, user_id=current_user["id"])
    from workflows.effgen.briefing import run_briefing
    background_tasks.add_task(run_briefing, job_id)
    logger.info(f"job {job_id} queued  workflow=briefing")
    return JobCreatedResponse(job_id=job_id)


@router.post("/rag/ingest", response_model=JobCreatedResponse)
async def trigger_rag_ingest(
    request: RagIngestRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
) -> JobCreatedResponse:
    job_id = await create_job("rag-ingest", request.model_dump(), user_id=current_user["id"])
    from workflows.effgen.rag_ingest import run_rag_ingest
    background_tasks.add_task(run_rag_ingest, job_id, request)
    logger.info(f"job {job_id} queued  workflow=rag-ingest  collection={request.collection_id!r}")
    return JobCreatedResponse(job_id=job_id)


@router.post("/rag/chat")
async def rag_chat(
    request: RagChatRequest,
    req: Request,
    current_user: dict = Depends(get_current_user),
):
    from workflows.effgen.rag_chat import stream_rag_chat

    accept = req.headers.get("accept", "")
    if "text/event-stream" in accept:
        return StreamingResponse(
            stream_rag_chat(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    else:
        content = ""
        async for chunk in stream_rag_chat(request):
            if chunk.startswith("data: ") and "[DONE]" not in chunk and "[ERROR]" not in chunk:
                content += chunk[len("data: "):].rstrip("\n")
        return JSONResponse({"content": content})


@router.post("/learning-plan", response_model=JobCreatedResponse)
async def trigger_learning_plan(
    request: LearningPlanRequest, background_tasks: BackgroundTasks
) -> JobCreatedResponse:
    # No auth — called service-to-service from kids-learning (no user token available)
    job_id = await create_job("learning-plan", request.model_dump())
    from workflows.effgen.learning_plan import run_learning_plan
    background_tasks.add_task(run_learning_plan, job_id, request)
    logger.info(f"job {job_id} queued  workflow=learning-plan  child={request.child.name!r}")
    return JobCreatedResponse(job_id=job_id)


@router.get("/learning-plan/{job_id}", response_model=JobStatusResponse)
async def get_learning_plan_status(job_id: str) -> JobStatusResponse:
    # No auth — polled service-to-service from kids-learning
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JobStatusResponse(**job)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_workflow_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
) -> JobStatusResponse:
    job = await get_job(job_id, user_id=current_user["id"])
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JobStatusResponse(**job)


@router.post("/soccer-analyst", response_model=SoccerAnalystResponse)
async def soccer_analyst(request: SoccerAnalystRequest) -> SoccerAnalystResponse:
    import routers.workflows as _self
    if not hasattr(_self, "run_soccer_analyst"):
        from workflows.soccer.analyst import run_soccer_analyst as _fn
        _self.run_soccer_analyst = _fn
    try:
        return await _self.run_soccer_analyst(request)
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=422, content={"error": str(e)})


@router.get("/", response_model=list[JobStatusResponse])
async def list_workflows(
    type: str | None = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
) -> list[JobStatusResponse]:
    jobs = await list_jobs(job_type=type, limit=limit, user_id=current_user["id"])
    return [JobStatusResponse(**j) for j in jobs]
