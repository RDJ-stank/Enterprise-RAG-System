import logging
from typing import Optional

import httpx

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
REQUEST_TIMEOUT = 60.0


class LLMClient:
    """DeepSeek API 异步客户端。

    封装与 DeepSeek API 的通信，使用 OpenAI 兼容接口 (/chat/completions)。
    内置重试机制和超时控制。

    生命周期：每次请求按需创建，或作为单例复用 AsyncClient。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """初始化 DeepSeek API 客户端。

        Args:
            api_key: API 密钥。默认从 config.DEEPSEEK_API_KEY 读取。
            base_url: API 基础地址。默认从 config.DEEPSEEK_BASE_URL 读取。
        """
        self._api_key = api_key or DEEPSEEK_API_KEY
        self._base_url = (base_url or DEEPSEEK_BASE_URL).rstrip("/")
        if not self._api_key:
            raise ValueError(
                "DeepSeek API Key 未配置，请设置环境变量 DEEPSEEK_API_KEY"
            )

    async def generate_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """调用 DeepSeek API 生成回答。

        Args:
            system_prompt: 系统指令，定义助手行为边界。
            user_prompt: 用户提示词（含检索上下文和用户问题）。
            model: 模型名称。默认使用 config.LLM_MODEL (deepseek-chat)。
            temperature: 生成温度。默认使用 config.LLM_TEMPERATURE (0.3)。
            max_tokens: 最大生成 token 数。

        Returns:
            DeepSeek 生成的回答文本。

        Raises:
            httpx.HTTPStatusError: API 返回非 2xx 状态码。
            httpx.TimeoutException: 请求超时。
            RuntimeError: 所有重试均失败或 API 返回了意外的响应结构。
        """
        resolved_model = model or LLM_MODEL
        resolved_temp = temperature if temperature is not None else LLM_TEMPERATURE

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "model": resolved_model,
            "messages": messages,
            "temperature": resolved_temp,
            "max_tokens": max_tokens,
        }

        last_error: Optional[Exception] = None

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        ) as client:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    logger.info(
                        "DeepSeek API 请求 (attempt %d/%d): model=%s, prompt_len=%d",
                        attempt + 1,
                        MAX_RETRIES + 1,
                        resolved_model,
                        len(user_prompt),
                    )
                    resp = await client.post(
                        "/v1/chat/completions",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    choices = data.get("choices", [])
                    if not choices:
                        raise RuntimeError("API 返回了空的 choices 数组")

                    answer: str = (
                        choices[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    if not answer:
                        logger.warning("API 返回了空的 content")
                        return "（模型未生成有效回答，请稍后重试）"

                    logger.info("DeepSeek API 响应成功，answer_len=%d", len(answer))
                    return answer

                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    status = exc.response.status_code
                    if status == 429 and attempt < MAX_RETRIES:
                        logger.warning("API 限流 (429)，将重试...")
                        continue
                    if status >= 500 and attempt < MAX_RETRIES:
                        logger.warning("API 服务器错误 (%d)，将重试...", status)
                        continue
                    logger.exception("DeepSeek API 请求失败 (status=%d)", status)
                    raise

                except httpx.TimeoutException as exc:
                    last_error = exc
                    if attempt < MAX_RETRIES:
                        logger.warning("API 请求超时，将重试...")
                        continue
                    logger.exception("DeepSeek API 请求超时，已达最大重试次数")
                    raise

                except Exception as exc:
                    last_error = exc
                    if attempt < MAX_RETRIES:
                        logger.warning("API 请求异常: %s，将重试...", exc)
                        continue
                    logger.exception("DeepSeek API 请求异常，已达最大重试次数")
                    raise

        raise RuntimeError(
            f"DeepSeek API 调用失败，已重试 {MAX_RETRIES} 次"
        ) from last_error
