"""LLM 服务。

统一的 LLM 调用接口，支持 OpenAI / Azure / Ollama / DeepSeek。
使用 LangChain 的 init_chat_model 实现多 Provider 切换。
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class LLMService:
    """LLM 调用服务。

    提供同步和异步的 LLM 调用接口。

    Example:
        >>> llm = LLMService(provider="openai", model="gpt-4o", api_key="sk-...")
        >>> answer = llm.invoke("什么是机器学习？")
        >>> answer = await llm.ainvoke("什么是机器学习？")
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._llm = None  # 懒加载

    def _get_llm(self):
        """懒加载 LLM 实例。"""
        if self._llm is None:
            try:
                from langchain.chat_models import init_chat_model

                # 构建模型参数
                kwargs = {
                    "model": self._model,
                    "model_provider": self._provider,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                }
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                if self._base_url:
                    kwargs["base_url"] = self._base_url

                # Qwen3 模型默认开启思考模式，需要通过 extra_body 关闭
                if "qwen3" in self._model.lower():
                    kwargs["extra_body"] = {"enable_thinking": False}

                self._llm = init_chat_model(**kwargs)
                logger.info(f"Initialized LLM: {self._provider}/{self._model}")
            except ImportError:
                raise ImportError(
                    "langchain is required. Install with: pip install langchain"
                )
        return self._llm

    def invoke(self, prompt: str, system_prompt: str = "") -> str:
        """同步调用 LLM。

        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）

        Returns:
            LLM 生成的文本
        """
        llm = self._get_llm()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = llm.invoke(messages)
        return response.content

    async def ainvoke(self, prompt: str, system_prompt: str = "") -> str:
        """异步调用 LLM。

        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）

        Returns:
            LLM 生成的文本
        """
        llm = self._get_llm()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await llm.ainvoke(messages)
        return response.content

    @staticmethod
    def from_settings(settings) -> "LLMService":
        """从 Settings 对象创建 LLMService。"""
        return LLMService(
            provider=settings.llm.provider,
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )
