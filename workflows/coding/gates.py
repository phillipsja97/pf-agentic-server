import fnmatch
import re
import subprocess
from pathlib import Path

from core.logging import logger
from workflows.coding.brain import brain_dir, project_dir


_DEFAULT_PROTECTED_PATTERNS = [
    ".env", "*.pem", "*.key", "*.p12", "*.pfx",
    "credentials*", "secrets*", "id_rsa", "id_ed25519",
]

_CI_PASS_STATUSES = {"pass", "skip", "success", ""}


def run_gate_check(slug: str, spec_n: int, pr_url: str) -> None:
    """Run all gates against a PR. Raises RuntimeError on any hard failure."""
    pr_match = re.search(r"/pull/(\d+)", pr_url)
    repo_match = re.search(r"github\.com/([^/]+/[^/]+)/pull", pr_url)
    if not pr_match or not repo_match:
        raise RuntimeError(f"cannot parse PR URL: {pr_url}")
    pr_number = pr_match.group(1)
    repo = repo_match.group(1)

    proj_dir = project_dir(slug)

    _gate_secret_scan(proj_dir)
    _gate_protected_files(pr_number, repo, proj_dir)
    _gate_ci_checks(pr_number, repo)
    _gate_spec_acceptance(pr_number, repo, slug)


def _gate_secret_scan(proj_dir: Path) -> None:
    try:
        result = subprocess.run(
            ["gitleaks", "detect", "--source", str(proj_dir), "--no-git", "--redact"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        logger.warning("gitleaks not installed — skipping secret scan gate")
        return
    if result.returncode == 0:
        return
    summary_lines = [
        l for l in result.stdout.splitlines()
        if any(w in l.lower() for w in ("leak", "finding", "secret", "wrn", "err"))
    ]
    summary = "\n".join(summary_lines)[-300:] or result.stdout[-300:]
    raise RuntimeError(f"GATE_FAIL secret_scan: {summary}")


def _gate_protected_files(pr_number: str, repo: str, proj_dir: Path) -> None:
    result = subprocess.run(
        ["gh", "pr", "diff", pr_number, "--name-only", "--repo", repo],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GATE_FAIL protected_files: could not retrieve PR diff: {result.stderr.strip()}")

    changed = [f for f in result.stdout.strip().splitlines() if f]
    protect_file = proj_dir / ".hermes-protect"
    patterns = (
        [l.strip() for l in protect_file.read_text().splitlines()]
        if protect_file.exists()
        else _DEFAULT_PROTECTED_PATTERNS
    )

    for pattern in patterns:
        if not pattern or pattern.startswith("#"):
            continue
        for filepath in changed:
            if fnmatch.fnmatch(filepath, pattern) or fnmatch.fnmatch(Path(filepath).name, pattern):
                raise RuntimeError(f"GATE_FAIL protected_files: {filepath} matches protected pattern {pattern}")


def _gate_ci_checks(pr_number: str, repo: str) -> None:
    result = subprocess.run(
        ["gh", "pr", "checks", pr_number, "--repo", repo],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        combined = (result.stdout + result.stderr).lower()
        if "no checks" in combined:
            logger.info("Gate 3: no CI checks configured — passing automatically")
            return
        raise RuntimeError(f"GATE_FAIL tests: could not retrieve PR checks: {result.stderr.strip()}")

    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            status = parts[1].strip().lower()
            if status not in _CI_PASS_STATUSES:
                raise RuntimeError(f"GATE_FAIL tests: check '{parts[0].strip()}' is {status}")


def _gate_spec_acceptance(pr_number: str, repo: str, slug: str) -> None:
    active_spec_path = brain_dir(slug) / "active-spec.md"
    if not active_spec_path.exists():
        logger.info("Gate 4: no active-spec.md — passing automatically")
        return

    criteria_lines: list[str] = []
    in_criteria = False
    for line in active_spec_path.read_text().splitlines():
        if re.match(r"^#{1,3}\s*Acceptance Criteria", line, re.IGNORECASE):
            in_criteria = True
            continue
        if in_criteria:
            if re.match(r"^#{1,3}\s", line):
                break
            if line.strip():
                criteria_lines.append(line.strip().lstrip("-* "))

    if not criteria_lines:
        logger.info("Gate 4: no acceptance criteria found — passing automatically")
        return

    result = subprocess.run(
        ["gh", "pr", "view", pr_number, "--repo", repo, "--json", "body", "--jq", '.body // ""'],
        capture_output=True, text=True,
    )
    pr_body = result.stdout.lower()

    if not pr_body.strip():
        logger.warning("Gate 4: PR body is empty — acceptance criteria not verified (warning only)")
        return

    _stopwords = {"that", "this", "with", "from", "have", "must", "will", "when", "should", "each"}
    missing: list[str] = []
    for criterion in criteria_lines:
        keywords = [w for w in criterion.lower().split() if len(w) > 4 and w not in _stopwords][:3]
        if keywords and not all(kw in pr_body for kw in keywords):
            missing.append(criterion)

    if len(missing) >= len(criteria_lines):
        logger.warning(
            f"Gate 4: PR body does not reference criteria (warning only): {'; '.join(missing[:3])}"
        )
