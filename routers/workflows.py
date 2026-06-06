from fastapi import APIRouter, BackgroundTasks, HTTPException

from core.logging import logger
from core.storage.jobs import create_job, get_job, list_jobs
from schemas.models import JobCreatedResponse, JobStatusResponse, ResearchRequest

router = APIRouter(tags=["workflows"])


@router.post("/research", response_model=JobCreatedResponse)
async def trigger_research(
    request: ResearchRequest, background_tasks: BackgroundTasks
) -> JobCreatedResponse:
    job_id = await create_job("research", request.model_dump())
    from workflows.effgen.research import run_research
    background_tasks.add_task(run_research, job_id, request)
    logger.info(f"job {job_id} queued  workflow=research  query={request.query!r}")
    return JobCreatedResponse(job_id=job_id)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_workflow_status(job_id: str) -> JobStatusResponse:
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JobStatusResponse(**job)


@router.get("/", response_model=list[JobStatusResponse])
async def list_workflows(type: str | None = None, limit: int = 50) -> list[JobStatusResponse]:
    jobs = await list_jobs(job_type=type, limit=limit)
    return [JobStatusResponse(**j) for j in jobs]
