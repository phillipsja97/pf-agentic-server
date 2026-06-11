import asyncio
import re
import subprocess
from typing import Optional

from pydantic import BaseModel

from config import settings
from core.logging import logger
from core.storage.jobs import update_job
from core.tracing import observe
from schemas.models import CodingRequest
from workflows.coding.brain import (
    advance_state, append_decision, append_inventory,
    brain_dir, init_brain, load_plan, load_prs, project_dir,
    record_pr, read_state, spec_status_file, write_active_spec,
)
from workflows.coding.claude_session import capture_pane, kill_session, send_keys, spawn_claude_session
from workflows.coding.gates import run_gate_check

POLL_INTERVAL = 30
MAX_SILENCE_SECONDS = 600
MAX_TOTAL_SECONDS = 3600


class SpecItem(BaseModel):
    title: str
    description: str
    acceptance_criteria: list[str]


class PlanOutput(BaseModel):
    project_name: str
    github_repo_description: str
    specs: list[SpecItem]


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


def _slugify(text: str) -> str:
    slug = text.lower()[:40]
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug or "project"


def _run_planner_sync(model, idea: str) -> PlanOutput:
    from effgen import create_agent

    agent = create_agent("research", model, tool_calling_mode="react")
    task = (
        f"Project idea: {idea}\n\n"
        "You are a senior software architect. Break this idea into an ordered sequence of implementable specs.\n"
        "Rules:\n"
        "- Each spec must be self-contained and buildable in a single coding session\n"
        "- Start with foundation (project setup, data models, core infrastructure) and work toward features\n"
        "- 3 to 8 specs for most projects\n"
        "- Each spec should take 1-4 hours of coding work\n"
        "- Write 2-4 clear, testable acceptance criteria per spec\n\n"
        "Return a structured plan with project_name, github_repo_description, and the ordered list of specs."
    )
    response = agent.run(task, output_model=PlanOutput)
    if not response.success:
        reason = (response.metadata or {}).get("reason", "unknown")
        raise RuntimeError(f"Planner failed: reason={reason}")
    parsed: Optional[PlanOutput] = (response.metadata or {}).get("parsed")
    if parsed is None:
        raise RuntimeError("Planner did not return structured output")
    return parsed


def _get_github_user() -> str:
    result = subprocess.run(
        ["gh", "api", "user", "--jq", ".login"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _repo_exists(slug: str, github_user: str) -> bool:
    result = subprocess.run(
        ["gh", "repo", "view", f"{github_user}/{slug}"],
        capture_output=True,
    )
    return result.returncode == 0


def _create_repo(slug: str, description: str, github_user: str) -> None:
    result = subprocess.run(
        ["gh", "repo", "create", f"{github_user}/{slug}",
         "--private", "--description", description],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh repo create failed: {result.stderr.strip()}")


def _clone_repo(slug: str, github_user: str) -> None:
    proj_path = project_dir(slug)
    proj_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["gh", "repo", "clone", f"{github_user}/{slug}", str(proj_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh repo clone failed: {result.stderr.strip()}")


def _claude_md_committed(slug: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(project_dir(slug)), "log", "--oneline", "-1", "--", "CLAUDE.md"],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def _write_project_claude_md(slug: str, project_name: str) -> None:
    proj_path = project_dir(slug)
    active_spec_path = brain_dir(slug) / "active-spec.md"
    content = f"""# {project_name}

This project is built by an automated spec-driven coding workflow.

## Completion Protocol

When your spec is complete:
1. Create the feature branch specified in the active spec
2. Commit all changes with a descriptive message
3. Open a PR against the default branch
4. Write the sentinel: `echo "PR_READY: <pr-url>" > /tmp/cc-{slug}-spec<N>-status` (use the correct spec number from the active spec)
5. Print: `PR_READY: <pr-url>`

If blocked: print `BLOCKED: <reason>`
If you need clarification: print `QUESTION: <question>`

Work autonomously. Do not ask for confirmation before taking actions.

## Active Spec

The current spec is always at: {active_spec_path}
"""
    (proj_path / "CLAUDE.md").write_text(content)


def _initial_commit(slug: str) -> None:
    proj_path = project_dir(slug)
    subprocess.run(["git", "-C", str(proj_path), "add", "CLAUDE.md"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(proj_path), "commit", "-m", "chore: add CLAUDE.md for coding agent"],
        check=True, capture_output=True,
    )
    subprocess.run(["git", "-C", str(proj_path), "push"], check=True, capture_output=True)


def _merge_pr(pr_url: str) -> None:
    pr_match = re.search(r"/pull/(\d+)", pr_url)
    repo_match = re.search(r"github\.com/([^/]+/[^/]+)/pull", pr_url)
    if not pr_match or not repo_match:
        raise RuntimeError(f"cannot parse PR URL: {pr_url}")
    result = subprocess.run(
        ["gh", "pr", "merge", pr_match.group(1), "--squash", "--delete-branch",
         "--repo", repo_match.group(1)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh pr merge failed: {result.stderr.strip()}")


async def _poll_for_pr(
    job_id: str, slug: str, spec_n: int, specs_total: int, specs_done: int
) -> str:
    status_file = spec_status_file(slug, spec_n)
    loop = asyncio.get_event_loop()
    start = loop.time()
    last_activity = start
    last_pane = ""

    while True:
        now = loop.time()
        if now - start > MAX_TOTAL_SECONDS:
            raise RuntimeError("max wait time exceeded (1 hour)")

        if status_file.exists():
            for line in status_file.read_text().splitlines():
                s = line.strip()
                if s.startswith("PR_READY:"):
                    match = re.search(r"PR_READY:\s*(https?://\S+)", s)
                    if match:
                        return match.group(1)
                elif s.startswith("BLOCKED:"):
                    reason = s[len("BLOCKED:"):].strip()
                    if reason and not reason.startswith("<"):
                        raise RuntimeError(f"BLOCKED: {reason}")

        pane = await asyncio.to_thread(capture_pane, slug, spec_n)

        for line in pane.splitlines():
            s = line.strip()
            if s.startswith("PR_READY:"):
                match = re.search(r"PR_READY:\s*(https?://\S+)", s)
                if match:
                    return match.group(1)
            elif s.startswith("BLOCKED:"):
                reason = s[len("BLOCKED:"):].strip()
                # Skip template placeholder text from our own prompt
                if reason and not reason.startswith("<"):
                    raise RuntimeError(f"BLOCKED: {reason}")
            elif s.startswith("QUESTION:"):
                question = s[len("QUESTION:"):].strip()
                if question and not question.startswith("<"):
                    raise RuntimeError(f"Needs human input — Claude asks: {question}")

        # Reset the silence timer whenever the pane content changes — Claude is working.
        if pane != last_pane:
            last_activity = now
            last_pane = pane

        if now - last_activity > MAX_SILENCE_SECONDS:
            await asyncio.to_thread(
                send_keys, slug, spec_n,
                "Are you still working? Print PR_READY: <url> when done, QUESTION: <question> if you need help, or BLOCKED: <reason> if stuck.",
            )
            last_activity = now

        last_lines = "\n".join(pane.strip().splitlines()[-3:])
        await update_job(job_id, "running", result={
            "phase": "building",
            "slug": slug,
            "current_spec": spec_n,
            "specs_total": specs_total,
            "specs_completed": specs_done,
            "last_output": last_lines,
        })

        await asyncio.sleep(POLL_INTERVAL)


@observe(name="coding-workflow")
async def run_app_builder(job_id: str, request: CodingRequest) -> None:
    logger.info(f"job {job_id} → running  workflow=coding  idea={request.idea[:80]!r}")

    try:
        # --- Phase 1: Plan (skip if resuming an existing slug) ---
        is_resume = bool(request.slug and (brain_dir(request.slug) / "plan.json").exists())

        if is_resume:
            slug = request.slug
            plan_data = load_plan(slug)
            specs = plan_data["specs"]
            project_name = plan_data["project_name"]
            repo_description = plan_data["github_repo_description"]
            logger.info(f"job {job_id} → resuming  slug={slug}")
            await update_job(job_id, "running", result={
                "phase": "setup", "slug": slug,
                "message": f"Resuming '{slug}'...",
            })
        else:
            await update_job(job_id, "running", result={
                "phase": "planning", "message": "Decomposing idea into specs...",
            })
            model = await _get_model()
            try:
                plan = await asyncio.to_thread(_run_planner_sync, model, request.idea)
            except Exception as e:
                await update_job(job_id, "failed", error=f"Planning failed: {e}")
                return

            slug = request.slug or _slugify(plan.project_name)
            specs = [s.model_dump() for s in plan.specs]
            project_name = plan.project_name
            repo_description = plan.github_repo_description

            logger.info(f"job {job_id} → plan ready  slug={slug}  specs={len(specs)}")
            await update_job(job_id, "running", result={
                "phase": "setup", "slug": slug,
                "specs_total": len(specs),
                "message": f"Creating project '{slug}' with {len(specs)} specs...",
            })
            init_brain(slug, project_name, repo_description, specs)

        specs_total = len(specs)

        # --- Phase 2: Repo setup (idempotent) ---
        try:
            github_user = await asyncio.to_thread(_get_github_user)

            if not await asyncio.to_thread(_repo_exists, slug, github_user):
                await asyncio.to_thread(_create_repo, slug, repo_description, github_user)

            if not project_dir(slug).exists():
                await asyncio.to_thread(_clone_repo, slug, github_user)

            if not await asyncio.to_thread(_claude_md_committed, slug):
                _write_project_claude_md(slug, project_name)
                await asyncio.to_thread(_initial_commit, slug)
        except Exception as e:
            await update_job(job_id, "failed", error=f"Repo setup failed: {e}")
            return

        # --- Phase 3: Build loop (resume from current_spec in state) ---
        state = read_state(slug)
        start_spec = int(state.get("current_spec", "1"))
        prior_prs = load_prs(slug)
        pr_urls: list[str] = list(prior_prs.values())

        for i, spec in enumerate(specs, 1):
            if i < start_spec:
                continue  # already merged in a previous run

            spec_n = i
            spec_title = spec["title"]

            await update_job(job_id, "running", result={
                "phase": "building",
                "slug": slug,
                "current_spec": spec_n,
                "specs_total": specs_total,
                "specs_completed": len(pr_urls),
                "last_output": f"Starting spec {spec_n}: {spec_title}",
                "pr_urls": pr_urls,
            })

            try:
                active_spec_path = write_active_spec(slug, spec_n)
            except Exception as e:
                await update_job(job_id, "failed", error=f"Spec {spec_n} write failed: {e}")
                return

            kill_session(slug, spec_n)
            spec_status_file(slug, spec_n).unlink(missing_ok=True)

            try:
                await asyncio.to_thread(
                    spawn_claude_session, slug, spec_n, project_dir(slug), active_spec_path
                )
            except Exception as e:
                await update_job(job_id, "failed", error=f"Spec {spec_n} session spawn failed: {e}")
                return

            try:
                pr_url = await _poll_for_pr(job_id, slug, spec_n, specs_total, len(pr_urls))
            except Exception as e:
                kill_session(slug, spec_n)
                await update_job(job_id, "failed", error=f"Spec {spec_n} polling failed: {e}")
                return
            finally:
                kill_session(slug, spec_n)

            await update_job(job_id, "running", result={
                "phase": "gate_check",
                "slug": slug,
                "current_spec": spec_n,
                "specs_total": specs_total,
                "specs_completed": len(pr_urls),
                "last_output": f"Running gate checks for spec {spec_n}...",
                "pr_urls": pr_urls,
            })

            try:
                await asyncio.to_thread(run_gate_check, slug, spec_n, pr_url)
            except Exception as e:
                await update_job(job_id, "failed", error=f"Spec {spec_n} gate check failed: {e}")
                return

            try:
                await asyncio.to_thread(_merge_pr, pr_url)
            except Exception as e:
                await update_job(job_id, "failed", error=f"Spec {spec_n} merge failed: {e}")
                return

            pr_urls.append(pr_url)
            record_pr(slug, spec_n, pr_url)
            advance_state(slug, spec_n, pr_url)
            append_decision(slug, f"Merged spec {spec_n}: {spec_title} — {pr_url}")
            append_inventory(slug, spec_n, spec_title)
            logger.info(f"job {job_id} → spec {spec_n}/{specs_total} complete  pr={pr_url}")

        await update_job(job_id, "completed", result={
            "slug": slug,
            "project_path": str(project_dir(slug)),
            "specs_completed": len(pr_urls),
            "pr_urls": pr_urls,
        })
        logger.info(f"job {job_id} → completed  workflow=coding  slug={slug}  specs={len(pr_urls)}")

    except Exception as e:
        logger.error(f"job {job_id} → failed  error={e}")
        await update_job(job_id, "failed", error=str(e))
