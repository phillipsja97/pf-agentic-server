import httpx
from core.logging import logger


async def send_push_notifications(
    tokens: list[str], title: str, body: str, job_id: str
) -> None:
    if not tokens:
        return
    messages = [
        {"to": token, "title": title, "body": body, "data": {"jobId": job_id}}
        for token in tokens
    ]
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                "https://exp.host/--/api/v2/push/send",
                json=messages,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
    except Exception as e:
        logger.warning(f"Push notification failed  job_id={job_id}  error={e}")
