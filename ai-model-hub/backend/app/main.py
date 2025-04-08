import os
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import json
from fastapi import FastAPI, HTTPException, Depends, Request, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy.orm import Session

# 導入服務和依賴
from app.services.api_key_service import APIKeyManager
from app.services.model_service import ModelService
from app.services.litellm_service import LiteLLMService
from app.services.cache_service import CacheService
from app.services.deep_search_service import DeepSearchService # 如果實現了
from app.services.analytics_service import AnalyticsService # 如果實現了
from app.services.personality_service import PersonalityService # 如果實現了

from app.db.database import get_db, init_db
from app.db.init_data import create_default_users # 如果實現了

from app.api import auth, models as api_models # 假設API路由放在app/api/

# 初始化數據庫 (通常在應用啟動時)
# init_db()

# 載入環境變數
load_dotenv()

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化應用程序
app = FastAPI(
    title="AI Model Hub API",
    description="API for accessing multiple AI models with comparison and blending capabilities",
    version="1.0.0"
)

# 設置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境中應限制來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服務
redis_url = os.getenv("REDIS_URL")
api_key_manager = APIKeyManager(redis_url=redis_url)
cache_service = CacheService(redis_url=redis_url)
model_service = ModelService()
personality_service = PersonalityService() # 初始化個性化服務

# 初始化AnalyticsService需要數據庫會話
# analytics_service = AnalyticsService(db=Depends(get_db)) # 這不能直接這樣用，需要在請求中注入

litellm_service = LiteLLMService(
    api_key_manager=api_key_manager,
    model_service=model_service,
    cache_service=cache_service,
    personality_service=personality_service # 傳遞個性化服務
)
# deep_search_service = DeepSearchService() # 如果實現了

# 包含認證路由
app.include_router(auth.router, prefix="/auth", tags=["authentication"])
# 可以添加其他API路由
# app.include_router(api_models.router, prefix="/models", tags=["models"])

# 數據模型 (可以移到 app/schemas.py)
class MessageSchema(BaseModel):
    role: str
    content: str

class CompletionRequest(BaseModel):
    model_ids: List[str]
    messages: List[MessageSchema]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False
    use_cache: Optional[bool] = True
    humanize: Optional[bool] = True

class DebateFormat(BaseModel):
    rounds: int = Field(default=3, ge=1, le=10)
    time_per_round: int = Field(default=500, ge=100, le=2000)
    moderation_level: str = Field(default="balanced", pattern="^(strict|balanced|open)$")
    topic_complexity: str = Field(default="medium", pattern="^(simple|medium|complex)$")
    debate_style: str = Field(default="argumentative",
                              pattern="^(argumentative|socratic|cooperative|persuasive)$")
    judge_criteria: List[str] = Field(default=["logic", "evidence", "clarity", "originality"])

class DebateRequest(BaseModel):
    model_ids: List[str]
    topic: str
    debate_format: Optional[DebateFormat] = DebateFormat()
    humanize: Optional[bool] = True

class BlendRequest(BaseModel):
    model_ids: List[str]
    messages: List[MessageSchema]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    blend_method: Optional[str] = "weighted"
    weights: Optional[Dict[str, float]] = None
    humanize: Optional[bool] = True

class DeepSearchRequest(BaseModel):
    query: str
    depth: Optional[int] = 1

# 背景任務函數
async def log_usage(db: Session, user_id: Optional[int], model_id: str, prompt_tokens: int, completion_tokens: int, latency: float, status: str, error_message: Optional[str]=None):
    """記錄模型使用情況到數據庫"""
    try:
        analytics_service = AnalyticsService(db) # 在任務中創建服務實例
        await analytics_service.record_model_performance(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_time=latency,
            status=status,
            error_message=error_message,
            user_id=user_id
        )
    except Exception as e:
        logger.error(f"Failed to log usage: {str(e)}")

# API路由
@app.on_event("startup")
async def startup_event():
    # 初始化數據庫表和默認用戶
    init_db()
    with SessionLocal() as db:
        create_default_users(db)
    logger.info("Application startup complete.")

@app.get("/")
async def root():
    return {"message": "Welcome to AI Model Hub API"}

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    # 檢查數據庫連接
    db_status = "ok"
    try:
        with SessionLocal() as db:
            db.execute("SELECT 1")
    except Exception as e:
        db_status = f"error: {str(e)}"
        logger.error(f"Database health check failed: {str(e)}")

    # 檢查緩存連接
    cache_status = "ok" if cache_service.enabled else "disabled"
    if cache_service.enabled:
        try:
            await cache_service.redis_client.ping()
        except Exception as e:
            cache_status = f"error: {str(e)}"
            logger.error(f"Cache health check failed: {str(e)}")

    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "api": "ok",
            "cache": cache_status,
            "database": db_status
        }
    }

@app.get("/models")
async def get_models():
    """獲取所有可用模型的列表"""
    return {"models": model_service.get_all_models()}

@app.get("/models/top")
async def get_top_models(limit: int = 5):
    """獲取頂級模型"""
    return {"models": model_service.get_top_models(limit=limit)}

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """獲取特定模型的詳細信息"""
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return {"model": model}

@app.post("/completions")
async def create_completion(request: CompletionRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db), current_user: auth.User = Depends(auth.get_current_user)):
    """從單個或多個模型獲取回應"""
    try:
        messages = [msg.dict() for msg in request.messages]
        start_time = time.time()
        user_id = current_user.id if current_user else None

        if len(request.model_ids) == 1:
            model_id = request.model_ids[0]
            try:
                response = await litellm_service.call_model(
                    model_id=model_id,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream=request.stream,
                    use_cache=request.use_cache,
                    humanize=request.humanize,
                    user_id=str(user_id) # 個性化服務需要字符串ID
                )
                status = "success"
                error_msg = None
                # 假設能從響應獲取token信息
                prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else 0
                completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0
            except Exception as e:
                logger.error(f"Error calling model {model_id}: {str(e)}")
                status = "error"
                error_msg = str(e)
                response = {"error": error_msg}
                prompt_tokens = sum(len(m['content'].split()) for m in messages)
                completion_tokens = 0

            end_time = time.time()
            latency = end_time - start_time

            # 異步記錄使用情況
            background_tasks.add_task(
                log_usage,
                db, user_id, model_id, prompt_tokens, completion_tokens, latency, status, error_msg
            )

            if status == "error":
                 raise HTTPException(status_code=500, detail=error_msg)

            return {"response": response}

        else:
            # 多模型調用
            responses = await litellm_service.call_multiple_models(
                model_ids=request.model_ids,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                use_cache=request.use_cache
            )

            end_time = time.time()
            total_latency = end_time - start_time
            logger.info(f"Multi-model completion completed in {total_latency:.2f}s")

            # 為每個成功/失敗的模型記錄使用情況
            for model_id, res in responses.items():
                if "error" not in res:
                    status = "success"
                    error_msg = None
                    prompt_tokens = res.usage.prompt_tokens if hasattr(res, 'usage') else 0
                    completion_tokens = res.usage.completion_tokens if hasattr(res, 'usage') else 0
                else:
                    status = "error"
                    error_msg = res["error"]
                    prompt_tokens = sum(len(m['content'].split()) for m in messages)
                    completion_tokens = 0

                background_tasks.add_task(
                    log_usage,
                    db, user_id, model_id, prompt_tokens, completion_tokens, 0, status, error_msg
                )

            return {"responses": responses}

    except Exception as e:
        logger.error(f"Error in completion endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debate")
async def create_debate(request: DebateRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db), current_user: auth.User = Depends(auth.get_current_user)):
    """創建模型間的辯論"""
    try:
        user_id = current_user.id if current_user else None
        start_time = time.time()

        debate_history = await litellm_service.debate_between_models(
            model_ids=request.model_ids,
            topic=request.topic,
            debate_format=request.debate_format,
            user_id=user_id,
            humanize=request.humanize
        )

        end_time = time.time()
        total_latency = end_time - start_time
        avg_latency_per_model = total_latency / (len(request.model_ids) * request.debate_format.rounds) if request.debate_format.rounds > 0 else 0

        # 對每個參與辯論的模型記錄總體使用情況
        for model_id in request.model_ids:
            background_tasks.add_task(
                log_usage,
                db, user_id, model_id, 0, 0, avg_latency_per_model, "success", "Debate participation"
            )

        return {"debate": debate_history}

    except Exception as e:
        logger.error(f"Error in debate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/blend")
async def blend_responses(request: BlendRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db), current_user: auth.User = Depends(auth.get_current_user)):
    """混合多個模型的回應"""
    try:
        user_id = current_user.id if current_user else None
        messages = [msg.dict() for msg in request.messages]
        start_time = time.time()

        blended_response = await litellm_service.blend_model_responses(
            model_ids=request.model_ids,
            messages=messages,
            blend_method=request.blend_method,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            weights=request.weights
            # 注意：人性化混合在混合策略內部處理
        )

        end_time = time.time()
        total_latency = end_time - start_time
        avg_latency_per_model = total_latency / len(request.model_ids) if len(request.model_ids) > 0 else 0

        # 對每個參與混合的模型記錄使用情況
        for model_id in request.model_ids:
            background_tasks.add_task(
                log_usage,
                db, user_id, model_id, 0, 0, avg_latency_per_model, "success", "Blend participation"
            )

        return {"response": blended_response}

    except Exception as e:
        logger.error(f"Error in blending endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 可以添加 /deep-search 和 /analytics 等其他路由

# 如果 main.py 是直接運行的腳本
if __name__ == "__main__":
    import uvicorn
    # 在開發模式下運行
    uvicorn.run("app.main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)), reload=True) 