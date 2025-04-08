import os
import time
import logging
from typing import Dict, List, Optional
import redis
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class APIKeyManager:
    def __init__(self, redis_url: str = None):
        # 嘗試連接 Redis，增加錯誤處理
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                # 測試連接
                self.redis_client.ping()
                logger.info("Redis connection successful")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {str(e)}. Falling back to in-memory storage.")
                self.redis_client = None

        self.openrouter_keys = self._load_openrouter_keys()
        self.provider_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "grok": os.getenv("GROK2_1212_API_KEY"),
            "google": os.getenv("GOOGLE_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
            "nvidia": os.getenv("NVIDIA_API_KEY"),
        }
        self.key_usage = {}  # 追蹤密鑰使用情況（內存版本）

    def _load_openrouter_keys(self) -> List[str]:
        """載入所有OpenRouter API密鑰"""
        keys = []
        for i in range(1, 14):  # 1到13的密鑰
            key = os.getenv(f"OPENROUTER_API_KEY_{i}")
            if key:
                keys.append(key)

        # 確保至少有一個可用密鑰
        if not keys:
            logger.warning("No OpenRouter API keys found in environment variables")
        else:
            logger.info(f"Loaded {len(keys)} OpenRouter API keys")

        return keys

    def get_openrouter_key(self) -> str:
        """取得下一個可用的OpenRouter密鑰，包含負載均衡策略"""
        now = datetime.now()
        available_keys = []

        # 使用 Redis 進行持久化狀態管理
        if self.redis_client:
            try:
                # 使用 Redis pipeline 確保原子性操作
                pipe = self.redis_client.pipeline()

                # 檢查所有密鑰的使用情況
                for i, key in enumerate(self.openrouter_keys):
                    key_id = f"openrouter_key_{i}"
                    # 使用 pipeline 批量獲取計數和上次重置時間
                    pipe.get(key_id)
                    pipe.get(f"{key_id}_last_reset")

                # 執行 pipeline
                results = pipe.execute()

                for i, key in enumerate(self.openrouter_keys):
                    usage_count_str = results[i*2]
                    last_reset_str = results[i*2 + 1]

                    usage_count = int(usage_count_str) if usage_count_str is not None else None
                    last_reset_ts = float(last_reset_str) if last_reset_str is not None else None

                    # 如果沒有使用記錄或需要重置
                    if usage_count is None or last_reset_ts is None:
                        available_keys.append((i, key, 0))
                        continue

                    # 檢查是否需要重置（每天重置）
                    last_reset_time = datetime.fromtimestamp(last_reset_ts)
                    if (now - last_reset_time).days >= 1:
                        available_keys.append((i, key, 0))
                        continue

                    # 檢查是否還有剩餘配額
                    if usage_count < 195:  # 留一些餘量避免超限
                        available_keys.append((i, key, usage_count))

                # 如果有可用密鑰，按使用量排序後選擇最少使用的
                if available_keys:
                    available_keys.sort(key=lambda x: x[2])  # 按使用量排序
                    selected_idx, selected_key, current_count = available_keys[0]

                    # 更新使用計數或初始化
                    key_id = f"openrouter_key_{selected_idx}"
                    pipe = self.redis_client.pipeline()
                    if current_count == 0:
                        pipe.set(key_id, 1)
                        pipe.set(f"{key_id}_last_reset", now.timestamp())
                    else:
                        pipe.incr(key_id)

                    pipe.execute()
                    return selected_key

                # 所有密鑰都已到達限制，隨機使用一個（緊急情況）
                logger.warning("All OpenRouter API keys have reached their daily limit! Using random key as fallback.")
                if not self.openrouter_keys:
                     raise ValueError("No OpenRouter keys available")
                return random.choice(self.openrouter_keys)

            except redis.RedisError as e:
                logger.error(f"Redis error in get_openrouter_key: {str(e)}. Falling back to in-memory key selection.")
                # 發生錯誤時回退到內存版本
            except Exception as e:
                 logger.error(f"Unexpected error in get_openrouter_key (Redis path): {str(e)}")
                 # 發生錯誤時回退到內存版本

        # 使用內存跟踪（Redis不可用的情況）
        logger.debug("Using in-memory key selection.")
        available_keys_mem = []
        for i, key in enumerate(self.openrouter_keys):
            key_id = f"openrouter_key_{i}"

            if key_id not in self.key_usage:
                self.key_usage[key_id] = {"count": 0, "last_reset": now}
                available_keys_mem.append((i, key, 0))
                continue

            # 檢查是否需要重置
            if (now - self.key_usage[key_id]["last_reset"]).days >= 1:
                self.key_usage[key_id] = {"count": 0, "last_reset": now}
                available_keys_mem.append((i, key, 0))
                continue

            # 檢查是否還有剩餘配額
            if self.key_usage[key_id]["count"] < 195:
                available_keys_mem.append((i, key, self.key_usage[key_id]["count"]))

        # 如果有可用密鑰，按使用量排序後選擇
        if available_keys_mem:
            available_keys_mem.sort(key=lambda x: x[2])  # 按使用量排序
            selected_idx, selected_key, _ = available_keys_mem[0]

            # 增加使用計數
            key_id = f"openrouter_key_{selected_idx}"
            self.key_usage[key_id]["count"] += 1
            logger.debug(f"Using OpenRouter key {selected_idx}, count: {self.key_usage[key_id]['count']}")
            return selected_key

        # 所有密鑰都已到達限制
        logger.warning("All OpenRouter API keys have reached their daily limit in memory tracker! Using random key.")
        if not self.openrouter_keys:
            raise ValueError("No OpenRouter keys available")
        return random.choice(self.openrouter_keys)

    def get_provider_key(self, provider: str) -> Optional[str]:
        """取得特定提供商的API密鑰"""
        key = self.provider_keys.get(provider)
        if not key:
            logger.warning(f"No API key found for provider: {provider}")
        return key

    def get_keys_status(self) -> List[Dict[str, Any]]:
        """獲取所有 OpenRouter 密鑰的使用狀態"""
        status = []
        now = datetime.now()

        if self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                for i, key in enumerate(self.openrouter_keys):
                    key_id = f"openrouter_key_{i}"
                    pipe.get(key_id)
                    pipe.get(f"{key_id}_last_reset")
                results = pipe.execute()

                for i, key in enumerate(self.openrouter_keys):
                    count_str = results[i*2]
                    reset_str = results[i*2 + 1]
                    count = int(count_str) if count_str is not None else 0
                    last_reset = None
                    if reset_str is not None:
                        last_reset_ts = float(reset_str)
                        last_reset = datetime.fromtimestamp(last_reset_ts)
                        # 檢查是否需要重置
                        if (now - last_reset).days >= 1:
                            count = 0 # 視為已重置

                    status.append({
                        "key_id": i,
                        "masked_key": key[:8] + "..." + key[-4:],
                        "count": count,
                        "last_reset": last_reset.isoformat() if last_reset else None,
                        "remaining": 200 - count,
                        "status": "active" if count < 195 else "limited"
                    })
            except Exception as e:
                logger.error(f"Error getting Redis key status: {str(e)}")
                # 回退到內存狀態
                for i, key in enumerate(self.openrouter_keys):
                    # ... (添加內存狀態獲取邏輯) ...
                    pass
        else:
            # 從內存獲取狀態
            for i, key in enumerate(self.openrouter_keys):
                key_id = f"openrouter_key_{i}"
                count = 0
                last_reset = None
                if key_id in self.key_usage:
                    count = self.key_usage[key_id]["count"]
                    last_reset = self.key_usage[key_id]["last_reset"]
                    # 檢查是否需要重置
                    if (now - last_reset).days >= 1:
                        count = 0 # 視為已重置

                status.append({
                    "key_id": i,
                    "masked_key": key[:8] + "..." + key[-4:],
                    "count": count,
                    "last_reset": last_reset.isoformat() if last_reset else None,
                    "remaining": 200 - count,
                    "status": "active" if count < 195 else "limited"
                })
        return status 