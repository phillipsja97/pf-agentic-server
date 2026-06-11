import json
import re
from datetime import datetime, timezone
from pathlib import Path

from config import settings


def brain_dir(slug: str) -> Path:
    return Path(settings.coding_brain_path) / slug


def project_dir(slug: str) -> Path:
    return Path(settings.projects_path) / slug


def spec_status_file(slug: str, spec_n: int) -> Path:
    return Path(f"/tmp/cc-{slug}-spec{spec_n}-status")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def init_brain(slug: str, project_name: str, repo_description: str, specs: list[dict]) -> None:
    d = brain_dir(slug)
    d.mkdir(parents=True, exist_ok=True)

    (d / "plan.json").write_text(json.dumps({
        "project_name": project_name,
        "github_repo_description": repo_description,
        "specs": specs,
    }))

    lines = [f"# Specs — {project_name}", ""]
    for i, spec in enumerate(specs, 1):
        lines += [f"## Spec {i}: {spec['title']}", "", spec["description"], ""]
        if spec.get("acceptance_criteria"):
            lines += ["### Acceptance Criteria", ""]
            lines += [f"- {c}" for c in spec["acceptance_criteria"]]
            lines += [""]
    (d / "specs.md").write_text("\n".join(lines))

    (d / "state.md").write_text(
        "current_spec: 1\n"
        "specs_completed: 0\n"
        "last_merged_spec: none\n"
        "pr_status: none\n"
        "last_signal: none\n"
    )
    (d / "decisions.md").write_text(f"# Decisions — {project_name}\n\nCreated: {_now()}\n")
    (d / "inventory.md").write_text(f"# Inventory — {project_name}\n\n")
    (d / "prs.json").write_text("{}")


def load_plan(slug: str) -> dict:
    return json.loads((brain_dir(slug) / "plan.json").read_text())


def record_pr(slug: str, spec_n: int, pr_url: str) -> None:
    path = brain_dir(slug) / "prs.json"
    prs = json.loads(path.read_text()) if path.exists() else {}
    prs[str(spec_n)] = pr_url
    path.write_text(json.dumps(prs))


def load_prs(slug: str) -> dict[str, str]:
    path = brain_dir(slug) / "prs.json"
    return json.loads(path.read_text()) if path.exists() else {}


def read_state(slug: str) -> dict[str, str]:
    path = brain_dir(slug) / "state.md"
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ": " in line:
            k, _, v = line.partition(": ")
            result[k.strip()] = v.strip()
    return result


def advance_state(slug: str, spec_n: int, pr_url: str) -> None:
    path = brain_dir(slug) / "state.md"
    content = path.read_text()
    completed = int(read_state(slug).get("specs_completed", "0")) + 1
    content = re.sub(r"^specs_completed:.*", f"specs_completed: {completed}", content, flags=re.MULTILINE)
    content = re.sub(r"^last_merged_spec:.*", f"last_merged_spec: {spec_n}", content, flags=re.MULTILINE)
    content = re.sub(r"^current_spec:.*", f"current_spec: {spec_n + 1}", content, flags=re.MULTILINE)
    content = re.sub(r"^pr_status:.*", "pr_status: merged", content, flags=re.MULTILINE)
    content = re.sub(r"^last_signal:.*", f"last_signal: merged spec {spec_n}", content, flags=re.MULTILINE)
    path.write_text(content)


def append_decision(slug: str, message: str) -> None:
    path = brain_dir(slug) / "decisions.md"
    with path.open("a") as f:
        f.write(f"\n## {_now()} — {message}\n")


def append_inventory(slug: str, spec_n: int, title: str) -> None:
    path = brain_dir(slug) / "inventory.md"
    with path.open("a") as f:
        f.write(f"spec-{spec_n}: {title} — DONE\n")


def write_active_spec(slug: str, spec_n: int) -> Path:
    d = brain_dir(slug)
    specs_content = (d / "specs.md").read_text()

    pattern = rf"(##\s*Spec\s+{spec_n}[:\s].+?)(?=\n##\s*Spec\s+\d|\Z)"
    match = re.search(pattern, specs_content, re.DOTALL | re.IGNORECASE)
    spec_section = match.group(1).strip() if match else specs_content.strip()

    first_line = spec_section.splitlines()[0] if spec_section else f"Spec {spec_n}"
    title = re.sub(r"^##\s*Spec\s+\d+[:\s]*", "", first_line).strip()
    branch_name = f"spec-{spec_n}-{re.sub(r'[^a-z0-9-]', '-', title.lower())[:30].strip('-')}"

    decisions_path = d / "decisions.md"
    decisions_tail = "\n".join(decisions_path.read_text().splitlines()[-10:]) if decisions_path.exists() else ""
    inventory = (d / "inventory.md").read_text() if (d / "inventory.md").exists() else ""

    status_file = spec_status_file(slug, spec_n)
    proj_dir = project_dir(slug)

    active_spec = f"""# Active Spec — #{spec_n}

project_slug: {slug}
spec_number: {spec_n}
spec_title: {title}

## Spec Content

{spec_section}

## Recent Decisions (context)

{decisions_tail}

## Inventory (modules already built)

{inventory}

## Instructions for Claude Code

Implement this spec in {proj_dir}/. When done:
1. Create feature branch `{branch_name}` from the repo default branch (master/main). Always branch from the repo default — never from a previous spec branch.
2. Commit all changes.
3. Open a PR against the repo default branch.
4. Run: `echo "PR_READY: <pr-url>" > {status_file}` (replace <pr-url> with the actual URL)
5. Print `PR_READY: <pr-url>`

Print `QUESTION: <question>` if you need clarification. Print `BLOCKED: <reason>` if you cannot proceed.
"""
    active_path = d / "active-spec.md"
    active_path.write_text(active_spec)
    return active_path
