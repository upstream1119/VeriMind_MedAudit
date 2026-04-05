"""
VeriMind-Med FastAPI 应用入口
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.api.routes import router


# ── 日志配置 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    settings = get_settings()
    logger.info(f"🩺 {settings.APP_NAME} v{settings.APP_VERSION} 启动中...")
    logger.info(f"   LLM 供应商: {settings.LLM_PROVIDER}")
    logger.info(f"   生成模型: {settings.LLM_MODEL_GENERATOR}")
    logger.info(f"   审计模型: {settings.LLM_MODEL_JUDGE}")
    logger.info(f"   Trust-Score: α={settings.TRUST_ALPHA}, β={settings.TRUST_BETA}")
    logger.info(f"   门控阈值: τ_high={settings.TRUST_THRESHOLD_HIGH}, τ_low={settings.TRUST_THRESHOLD_LOW}")
    yield
    logger.info(f"🩺 {settings.APP_NAME} 已关闭")


def create_app() -> FastAPI:
    """应用工厂"""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="基于多智能体协同与证据链溯源的高可靠医药决策系统",
        lifespan=lifespan,
    )

    # ── CORS 中间件 ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 注册路由 ──
    app.include_router(router, prefix="/api")

    return app


# ── 应用实例 (uvicorn 入口) ──
app = create_app()
