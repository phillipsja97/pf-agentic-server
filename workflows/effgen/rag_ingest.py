import asyncio
from pathlib import Path

from config import settings
from core.logging import logger
from core.storage.jobs import update_job
from core.tracing import observe
from schemas.models import RagIngestRequest


def _ingest_sync(source: str, index_path: Path) -> int:
    from effgen.rag import DocumentIngester
    from effgen.tools.builtin import Retrieval

    index_path.parent.mkdir(parents=True, exist_ok=True)

    ingester = DocumentIngester(chunk_size=500, chunk_overlap=100, show_progress=False)
    chunks = ingester.ingest(source)
    if not chunks:
        raise RuntimeError(f"No content extracted from source: {source!r}")

    retrieval = Retrieval(index_path=str(index_path))
    docs = [{"content": c.content, "id": c.id, "metadata": c.metadata} for c in chunks]
    retrieval.add_documents(docs, chunk=False)
    retrieval.save_index()

    return len(chunks)


@observe(name="rag-ingest-workflow")
async def run_rag_ingest(job_id: str, request: RagIngestRequest) -> None:
    logger.info(f"job {job_id} → running  workflow=rag-ingest  source={request.source!r}  collection={request.collection_id!r}")
    await update_job(job_id, "running")

    index_path = settings.brain_vault_path / "rag" / request.collection_id / "index.pkl"

    try:
        chunk_count = await asyncio.to_thread(_ingest_sync, request.source, index_path)

        result = {
            "collection_id": request.collection_id,
            "chunk_count": chunk_count,
            "index_path": str(index_path),
        }
        await update_job(job_id, "completed", result=result)
        logger.info(f"job {job_id} → completed  workflow=rag-ingest  chunks={chunk_count}")

    except Exception as e:
        logger.error(f"job {job_id} → failed  error={e}")
        await update_job(job_id, "failed", error=str(e))
