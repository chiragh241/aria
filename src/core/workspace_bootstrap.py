"""Workspace bootstrap: Q&A + LLM to generate SOUL.md, USER.md, IDENTITY.md, AGENTS.md."""

from __future__ import annotations

import re
from typing import Any

from ..utils.config import get_settings
from ..utils.logging import get_logger
from .workspace import (
    DEFAULT_TEMPLATES,
    WORKSPACE_FILES,
    is_bootstrap_needed,
    mark_bootstrap_done,
    write_workspace_file,
)

logger = get_logger(__name__)

BOOTSTRAP_PROMPT = """You are helping set up an AI assistant's workspace for Aria. Given the following short answers from the user, generate exactly four markdown files. Output ONLY the four blocks below, with no extra commentary. Use the exact section headers.

Format your response as:

<<SOUL.md>>
... markdown content for SOUL.md (personality, tone, how the assistant should behave) ...
<</SOUL.md>>

<<IDENTITY.md>>
... markdown content for IDENTITY.md (assistant name, role, how it presents itself) ...
<</IDENTITY.md>>

<<USER.md>>
... markdown content for USER.md (who the human user is: preferred name, what they care about, how they work) ...
<</USER.md>>

<<AGENTS.md>>
... markdown content for AGENTS.md (work habits, boundaries, when to ask vs act) ...
<</AGENTS.md>>

Keep each file concise (a few short paragraphs or bullet points). Write in first person for SOUL/IDENTITY (the assistant) and second/first for USER. For AGENTS use second person (how "you" should operate).

User answers:
{answers}
"""


def run_bootstrap_qa_cli() -> dict[str, str]:
    """Interactive CLI Q&A. Returns a dict of answers."""
    print("\n--- Aria workspace bootstrap ---\n")
    questions = [
        ("user_name", "What should the assistant call you? (preferred name)"),
        ("assistant_name", "What is your assistant's name? (e.g. Aria)"),
        ("assistant_role", "In one line: what is the assistant's role? (e.g. personal AI that gets things done)"),
        ("tone", "How should the assistant sound? (e.g. concise, friendly, no filler)"),
        ("work_habits", "Any work habits or boundaries? (e.g. ask before sending email, prefer shell over GUI)"),
    ]
    answers: dict[str, str] = {}
    for key, prompt in questions:
        try:
            answers[key] = input(f"  {prompt}\n  > ").strip() or "(not specified)"
        except EOFError:
            answers[key] = "(not specified)"
    return answers


def _format_answers_for_prompt(answers: dict[str, str]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in answers.items())


def _parse_llm_workspace_response(text: str) -> dict[str, str]:
    """Extract SOUL.md, IDENTITY.md, USER.md, AGENTS.md content from <<FILE>>...<</FILE>> blocks."""
    out: dict[str, str] = {}
    for name in WORKSPACE_FILES:
        pattern = rf"<<{re.escape(name)}>>\s*(.*?)\s*<</{re.escape(name)}>>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            out[name] = match.group(1).strip()
    return out


async def run_bootstrap_llm(llm_router: Any, answers: dict[str, str]) -> dict[str, str]:
    """
    Use LLM to generate the four workspace .md contents from Q&A answers.
    Returns dict mapping filename -> content. Empty dict on failure.
    """
    from .llm_router import LLMMessage

    prompt = BOOTSTRAP_PROMPT.format(answers=_format_answers_for_prompt(answers))
    messages = [LLMMessage(role="user", content=prompt)]
    try:
        response = await llm_router.generate(messages=messages, task_type="skill_creation")
        content = (response.content or "").strip()
        parsed = _parse_llm_workspace_response(content)
        if len(parsed) != 4:
            logger.warning("Bootstrap LLM did not return all four files", got=list(parsed.keys()))
        return parsed
    except Exception as e:
        logger.error("Bootstrap LLM failed", error=str(e))
        return {}


def run_bootstrap_template_only(answers: dict[str, str]) -> dict[str, str]:
    """Generate workspace files from answers using simple template substitution (no LLM)."""
    user_name = answers.get("user_name", "User")
    assistant_name = answers.get("assistant_name", get_settings().aria.name)
    role = answers.get("assistant_role", "personal AI assistant")
    tone = answers.get("tone", "concise and helpful")
    habits = answers.get("work_habits", "ask before external actions")

    return {
        "SOUL.md": (
            f"# Soul\n\n"
            f"I am {assistant_name}, a {role}. I am {tone}. "
            f"I get things done and learn how when I don't know. I don't refuse tasks; I use tools and figure it out.\n"
        ),
        "IDENTITY.md": (
            f"# Identity\n\n"
            f"Name: {assistant_name}. Role: {role}. "
            f"I present myself clearly and avoid filler. I have opinions and am genuinely helpful.\n"
        ),
        "USER.md": (
            f"# User\n\n"
            f"The user prefers to be called **{user_name}**. "
            f"Work habits / boundaries: {habits}\n"
        ),
        "AGENTS.md": (
            f"# Agents / Work habits\n\n"
            f"Operate with these boundaries: {habits}. "
            f"When in doubt about external actions, ask. Be thorough but concise.\n"
        ),
    }


async def run_bootstrap(
    llm_router: Any | None = None,
    use_llm: bool = True,
    answers: dict[str, str] | None = None,
) -> bool:
    """
    Run workspace bootstrap: Q&A (if answers not provided), then generate and write SOUL/IDENTITY/USER/AGENTS.
    If use_llm and llm_router is set, use LLM to generate content; else use template-only.
    Returns True if bootstrap ran and wrote files.
    """
    from .workspace import ensure_workspace_templates

    ensure_workspace_templates()
    if not is_bootstrap_needed():
        logger.info("Workspace already bootstrapped; skipping")
        return False

    if answers is None:
        answers = run_bootstrap_qa_cli()

    if use_llm and llm_router:
        contents = await run_bootstrap_llm(llm_router, answers)
        if not contents:
            logger.warning("Bootstrap LLM returned nothing; falling back to template")
            contents = run_bootstrap_template_only(answers)
    else:
        contents = run_bootstrap_template_only(answers)

    for name, content in contents.items():
        if name in WORKSPACE_FILES and content:
            write_workspace_file(name, content, user_id=None)
            logger.info("Wrote workspace file", file=name)

    mark_bootstrap_done()
    return True


def run_bootstrap_sync(
    llm_router: Any | None = None,
    use_llm: bool = True,
    answers: dict[str, str] | None = None,
) -> bool:
    """Synchronous wrapper for run_bootstrap (for CLI)."""
    import asyncio
    return asyncio.run(run_bootstrap(llm_router=llm_router, use_llm=use_llm, answers=answers))
