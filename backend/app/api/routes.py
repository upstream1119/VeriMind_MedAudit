"""
VeriMind-Med API 路由
"""

import time
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.config import get_settings
from app.models.schemas import (
    AuditQueryRequest,
    AuditQueryResponse,
    HealthResponse,
    TrustScoreDetail,
    TrustLevel,
    IntentType,
)
from app.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查 — 返回系统状态和当前模型配置"""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        llm_provider=settings.LLM_PROVIDER,
        models={
            "generator": settings.LLM_MODEL_GENERATOR,
            "judge": settings.LLM_MODEL_JUDGE,
            "router": settings.LLM_MODEL_ROUTER,
        },
    )


@router.post("/audit/query", response_model=AuditQueryResponse, tags=["审计"])
async def audit_query(request: AuditQueryRequest):
    """
    医药合规审计查询 (非流式)

    完整的 Agent DAG 流程:
    用户提问 → 知识对齐 → 意图路由 → 三粒度检索 → 推理生成 → 审计门控 → 返回
    """
    start_time = time.time()

    # TODO: 阶段三实现完整 Agent DAG, 当前返回模拟数据
    try:
        llm = get_llm_client()
        answer = await llm.generate(
            prompt=request.query,
            system_prompt="你是一个医药合规审计助手。请简要回答用户的医药相关问题。",
            model_role="generator",
        )

        processing_time = time.time() - start_time

        return AuditQueryResponse(
            query=request.query,
            normalized_query=request.query,
            intent=IntentType.CONCEPT,
            answer=answer,
            trust_score=TrustScoreDetail(
                s_ret=0.0,
                s_faith=0.0,
                w_authority=0.0,
                trust_score=0.0,
                trust_level=TrustLevel.WARNING,
            ),
            evidence=[],
            processing_time=round(processing_time, 3),
        )
    except Exception as e:
        logger.error(f"审计查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@router.post("/audit/query/stream", tags=["审计"])
async def audit_query_stream(request: AuditQueryRequest):
    """
    医药合规审计查询 (SSE 流式)

    前端通过 EventSource 接收逐 token 的流式回答
    """
    import json

    async def event_generator():
        try:
            llm = get_llm_client()

            # 发送开始事件
            yield f"data: {json.dumps({'type': 'start', 'query': request.query})}\n\n"

            # 流式生成回答
            full_answer = ""
            async for token in llm.generate_stream(
                prompt=request.query,
                system_prompt="你是一个医药合规审计助手。请简要回答用户的医药相关问题。",
                model_role="generator",
            ):
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # 发送完成事件 (含 Trust-Score)
            yield f"data: {json.dumps({'type': 'done', 'answer': full_answer})}\n\n"

        except Exception as e:
            logger.error(f"流式查询失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
