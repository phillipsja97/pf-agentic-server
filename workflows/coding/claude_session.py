import subprocess
import time
from pathlib import Path


def session_name(slug: str, spec_n: int) -> str:
    return f"cc-{slug}-spec{spec_n}"


def kill_session(slug: str, spec_n: int) -> None:
    subprocess.run(
        ["tmux", "kill-session", "-t", session_name(slug, spec_n)],
        capture_output=True,
    )


def capture_pane(slug: str, spec_n: int, lines: int = 2000) -> str:
    result = subprocess.run(
        ["tmux", "capture-pane", "-t", session_name(slug, spec_n), "-p", "-S", f"-{lines}"],
        capture_output=True, text=True,
    )
    return result.stdout


def send_keys(slug: str, spec_n: int, text: str) -> None:
    subprocess.run(
        ["tmux", "send-keys", "-t", session_name(slug, spec_n), text, "Enter"],
        capture_output=True,
    )


def spawn_claude_session(slug: str, spec_n: int, project_path: Path, active_spec_path: Path) -> None:
    session = session_name(slug, spec_n)

    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-x", "220", "-y", "50"],
        check=True,
    )

    subprocess.run(
        ["tmux", "send-keys", "-t", session,
         f"cd {project_path} && claude --dangerously-skip-permissions",
         "Enter"],
        check=True,
    )

    # Wait for Claude TUI to be ready (up to 60s)
    for _ in range(60):
        pane = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p"],
            capture_output=True, text=True,
        ).stdout
        if "bypass permissions on" in pane:
            break
        # Accept trust dialog if it appears for new projects
        if "Yes, I trust this folder" in pane:
            subprocess.run(["tmux", "send-keys", "-t", session, "", "Enter"])
        time.sleep(1)

    # Give the TUI input handler time to settle after the status bar appears
    time.sleep(3)

    prompt = (
        f"Read {active_spec_path} and implement the spec. "
        "Print PR_READY: <url> when the PR is open. "
        "Print QUESTION: <question> if you need clarification. "
        "Print BLOCKED: <reason> if you cannot proceed."
    )
    # Send text and Enter separately — sending together can cause Enter to arrive
    # before the input handler is live, leaving the prompt unsubmitted.
    subprocess.run(["tmux", "send-keys", "-t", session, prompt])
    time.sleep(1)
    subprocess.run(["tmux", "send-keys", "-t", session, "Enter"])
