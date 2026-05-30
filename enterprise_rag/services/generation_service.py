import logging

from infrastructure.llm_client import LLMClient

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """\
你是一个企业知识库助手。请严格基于以下提供的文档片段回答问题。
如果文档片段中没有足够信息，请明确说明"根据现有文档，无法回答该问题"，
不要编造任何信息。回答时请引用具体的文档来源。"""


class GenerationService:
    """生成编排服务。

    组装 RAG Prompt（System + Context + User），调用 LLM 生成最终答案。
    """

    @staticmethod
    async def generate(
        question: str,
        sources: list[dict],
    ) -> str:
        """基于检索结果生成回答。

        Args:
            question: 用户原始提问。
            sources: 检索到的文档片段列表，每项含 chunk_text, filename, score。

        Returns:
            LLM 生成的回答文本。
        """
        if not sources:
            return (
                "根据现有文档，无法回答该问题。\n\n"
                "可能的原因：\n"
                "1. 上传的文档为扫描件或图片，无法提取文字\n"
                "2. 知识库中尚无与问题相关的文档\n"
                "3. 文档尚未完成向量化入库\n\n"
                "建议：请确认上传的是包含可提取文字的 PDF 或 TXT 文件，"
                "并检查左侧文档列表中该文档的「块数」是否大于 0。"
            )

        context_parts = []
        for i, src in enumerate(sources, start=1):
            context_parts.append(
                f"[{i}] (来源: {src['filename']}, 相关度: {src['score']})\n{src['chunk_text']}"
            )
        context = "\n\n".join(context_parts)

        user_prompt = f"""参考文档片段:

{context}

用户问题: {question}

请根据上述参考文档片段，回答用户的问题。"""

        logger.info(
            "调用 LLM 生成: question_len=%d, sources=%d, context_len=%d",
            len(question),
            len(sources),
            len(user_prompt),
        )

        client = LLMClient()
        answer = await client.generate_answer(
            system_prompt=RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        return answer
