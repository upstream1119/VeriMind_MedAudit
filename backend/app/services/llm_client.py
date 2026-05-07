"""
VeriMind-Med 统一 LLM 客户端
基于 OpenAI 兼容协议封装, 上层 Agent 代码无需感知底层供应商
支持: 智谱 AI / DashScope / DeepSeek
"""

import logging
from functools import lru_cache
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from app.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    统一 LLM 调用客户端

    所有供应商均通过 OpenAI 兼容接口调用, 实现一次封装到处复用。
    支持普通生成和流式输出两种模式。

    Usage:
        client = LLMClient()
        response = await client.generate("你好")
        async for chunk in client.generate_stream("你好"):
            print(chunk, end="")
    """

    def __init__(self):
        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=settings.get_active_api_key(),
            base_url=settings.get_base_url(),
        )
        self._settings = settings
        logger.info(
            f"LLMClient 初始化完成 | 供应商: {settings.LLM_PROVIDER} "
            f"| 生成模型: {settings.LLM_MODEL_GENERATOR} "
            f"| 审计模型: {settings.LLM_MODEL_JUDGE}"
        )

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        model_role: str = "generator",
        temperature: float | None = None,
    ) -> str:
        """
        非流式生成

        Args:
            prompt: 用户输入
            system_prompt: 系统提示词
            model_role: 模型角色 (generator / judge / router), 决定使用哪个模型
            temperature: 覆盖默认温度
        """
        model = self._get_model(model_role)
        temp = temperature if temperature is not None else self._settings.LLM_TEMPERATURE

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
            )
            content = response.choices[0].message.content or ""
            logger.debug(f"LLM [{model_role}:{model}] 生成完成, 长度: {len(content)}")
            return content
        except Exception as e:
            logger.error(f"LLM 调用失败 [{model_role}:{model}]: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        model_role: str = "generator",
        temperature: float | None = None,
    ):
        """
        流式生成 (SSE 场景)

        Yields:
            每次产出一个 token 字符串
        """
        model = self._get_model(model_role)
        temp = temperature if temperature is not None else self._settings.LLM_TEMPERATURE

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LLM 流式调用失败 [{model_role}:{model}]: {e}")
            raise

    def _get_model(self, role: str) -> str:
        """根据角色返回对应的模型名称"""
        model_map = {
            "generator": self._settings.LLM_MODEL_GENERATOR,
            "judge": self._settings.LLM_MODEL_JUDGE,
            "router": self._settings.LLM_MODEL_ROUTER,
        }
        return model_map.get(role, self._settings.LLM_MODEL_GENERATOR)


# ── 依赖注入支持 ──
@lru_cache()
def get_llm_client() -> LLMClient:
    """获取原生的 LLM 客户端实例 (用于非 Agent 的直接调用)"""
    return LLMClient()

@lru_cache()
def get_langchain_client(role: str = "generator") -> ChatOpenAI:
    """获取供 LangGraph/LangChain 使用的 ChatOpenAI 实例"""
    settings = get_settings()
    
    model_map = {
        "generator": settings.LLM_MODEL_GENERATOR,
        "judge": settings.LLM_MODEL_JUDGE,
        "router": settings.LLM_MODEL_ROUTER,
    }
    model_name = model_map.get(role, settings.LLM_MODEL_GENERATOR)

    return ChatOpenAI(
        model=model_name,
        api_key=settings.get_active_api_key(),
        base_url=settings.get_base_url(),
        temperature=settings.LLM_TEMPERATURE,
        max_retries=2,
    )

def generate_structured_output(
    system_prompt: str,
    user_prompt: str,
    output_schema: type,
    role: str = "generator"
):
    """
    通用的结构化输出包裹函数
    利用 LangChain 的 with_structured_output 提取 JSON 化对象
    """
    llm = get_langchain_client(role=role)
    structured_llm = llm.with_structured_output(output_schema)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        result = structured_llm.invoke(messages)
        return result
    except Exception as e:
        logger.error(f"[LLM] 调用结构化输出失败: {e}")
        raise
