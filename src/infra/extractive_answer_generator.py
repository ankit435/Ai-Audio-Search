"""Default `AnswerGenerator` adapter (Stretch Goal) — no external LLM API
key required. Concatenates and cites the top retrieved chunk snippets
into a short extractive "answer", in speaker/timestamp order, so the
`POST /search/answer` endpoint always works offline (unlike a hosted-LLM
adapter, which this environment's corporate proxy likely blocks anyway
— see `openai_answer_generator.py` for that alternative, gated behind
an API key).
"""

from __future__ import annotations

from src.domain.models import Chunk

MAX_CHUNKS_IN_ANSWER = 3
MAX_SNIPPET_CHARS = 220


class ExtractiveAnswerGenerator:
    async def generate(self, query: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "No relevant results were found for this query."

        lines = [f'Based on the top {min(len(chunks), MAX_CHUNKS_IN_ANSWER)} matching passage(s) for "{query}":']
        for chunk in chunks[:MAX_CHUNKS_IN_ANSWER]:
            snippet = chunk.text if len(chunk.text) <= MAX_SNIPPET_CHARS else chunk.text[:MAX_SNIPPET_CHARS - 1].rstrip() + "…"
            lines.append(f"- [{chunk.speaker_id} @ {chunk.start_time:.1f}s] {snippet}")
        return "\n".join(lines)
