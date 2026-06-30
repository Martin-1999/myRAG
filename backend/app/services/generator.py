from __future__ import annotations

from langchain_openai import ChatOpenAI

from backend.app.core.config import get_settings


class AnswerGenerator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.llm_model_name,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_api_base,
            timeout=self.settings.llm_timeout_seconds,
        )

    async def generate(self, question: str, contexts: list[str]) -> tuple[str, str]:
        context_block = "\n\n".join(f"[Context {idx + 1}]\n{ctx}" for idx, ctx in enumerate(contexts))
        prompt = (
            "你是一个严谨的RAG问答助手。请仅基于给定上下文回答问题，"
            "如果上下文不足，请明确说明。\n\n"
            f"{context_block}\n\n"
            f"问题：{question}\n回答："
        )
        response = await self.llm.ainvoke(prompt)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        return answer, prompt
