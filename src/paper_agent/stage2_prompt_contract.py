"""Single frozen Stage 2 adjudication prompt and context-size proxy contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from typing import Protocol


ADJUDICATION_SYSTEM_PROMPT = (
    "Return only the required structured screening decision. "
    "Keep the rationale to one sentence of at most 20 words."
)
ADJUDICATION_USER_TEMPLATE = (
    "Query version: {query_version}\nQuery: {query}\nPaper ID: {paper_id}\n{document}"
)
OMLX_CHAT_INPUT_TOKEN_PROXY_ESTIMATOR = "omlx-chat-content-chars-proxy-v1"


class PromptPaper(Protocol):
    paper_id: str
    title: str
    abstract: str | None
    keywords: Sequence[str]


def render_stage2_document(paper: PromptPaper) -> str:
    """Render the text passed to both Stage 2 model paths."""

    return (
        f"Title: {paper.title}\nAbstract: {paper.abstract or ''}\n"
        f"Keywords: {', '.join(paper.keywords)}"
    )


def render_adjudication_user_prompt(
    *, query_version: str, query: str, paper: PromptPaper,
) -> str:
    return ADJUDICATION_USER_TEMPLATE.format(
        query_version=query_version,
        query=query,
        paper_id=paper.paper_id,
        document=render_stage2_document(paper),
    )


def adjudication_messages(
    *, query_version: str, query: str, paper: PromptPaper,
) -> tuple[dict[str, str], ...]:
    return (
        {"role": "system", "content": ADJUDICATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": render_adjudication_user_prompt(
                query_version=query_version, query=query, paper=paper,
            ),
        },
    )


def estimate_omlx_chat_input_token_proxy(
    messages: Sequence[Mapping[str, str]],
) -> int:
    """Return a stable chars/4 proxy bound, not an exact tokenizer count."""

    return sum(len(message.get("content", "")) for message in messages) // 4 + 1


def stage2_prompt_hash(prompt_version: str) -> str:
    encoded = json.dumps(
        {
            "version": prompt_version,
            "system": ADJUDICATION_SYSTEM_PROMPT,
            "user_template": ADJUDICATION_USER_TEMPLATE,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
