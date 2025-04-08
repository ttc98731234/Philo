import os
import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional, Union
import litellm
from litellm import acompletion
import re # 用於句子分割

from app.services.api_key_service import APIKeyManager
from app.services.model_service import ModelService, ModelInfo
from app.services.cache_service import CacheService
from app.services.personality_service import PersonalityService # 導入個性化服務
from app.api.auth import DebateFormat # 導入辯論格式模型

logger = logging.getLogger(__name__)

class LiteLLMService:
    def __init__(
        self,
        api_key_manager: APIKeyManager,
        model_service: ModelService,
        cache_service: Optional[CacheService] = None,
        personality_service: Optional[PersonalityService] = None, # 接收個性化服務
        analytics_service = None # 接收分析服務
    ):
        self.api_key_manager = api_key_manager
        self.model_service = model_service
        self.cache_service = cache_service
        self.personality_service = personality_service or PersonalityService() # 如果未提供，則創建默認實例
        self.analytics_service = analytics_service # 存儲分析服務

        # 配置LiteLLM
        litellm.telemetry = False # 禁用遙測
        litellm.num_retries = 3
        litellm.request_timeout = 120  # 秒

        # 設置模型映射規則
        self._setup_model_mapping()

    def _setup_model_mapping(self):
        """設置模型映射配置"""
        # OpenRouter映射
        litellm.model_alias_map = {
            "llama-4-m": "openrouter/meta/llama-4-m",
            "deepseek-v3": "openrouter/deepseek/deepseek-v3",
            "deepseek-v3-0324": "openrouter/deepseek/deepseek-v3-0324",
            "qwq-32b": "openrouter/nousresearch/qwq-32b",
            "perxparity": "openrouter/perplexity/perxparity",
            "gemini-2.5-pro-exp-03-25": "gemini/gemini-2.5-pro-exp-03-25",
            "gemini-2.0-pro-exp-02-05": "gemini/gemini-2.0-pro-exp-02-05",
            "gemini-2.0-flash-thinking-exp-01-21": "gemini/gemini-2.0-flash-thinking-exp-01-21",
            "gemini-2.0-flash-001": "gemini/gemini-2.0-flash-001",
            "gemini-2.0-flash-lite": "gemini/gemini-2.0-flash-lite",
            "gemini-1.5-pro-002": "gemini/gemini-1.5-pro-002",
            "deepseek-r1": "nvidia/deepseek-r1",
            "gemma-3-27b-it": "nvidia/gemma-3-27b-it",
            "llama-3.3-nemotron-super-49b-v1": "nvidia/llama-3.3-nemotron-super-49b-v1",
            "grok2-1212": "grok/grok2-1212",
            # 可以繼續添加其他提供商和模型的映射
        }
        # 增加重試和回退邏輯
        litellm.set_verbose = False # 生產環境關閉詳細日誌

    async def _call_model_with_retry(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        stream: bool,
        use_cache: bool,
        humanize: bool,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """內部調用單個模型的方法，包含重試邏輯"""
        # 如果需要人性化回應，添加個性化系統提示
        final_messages = messages.copy()
        if humanize and final_messages:
            personality_prompt = self.personality_service.get_personality_prompt(user_id)
            if final_messages[0].get("role") == "system":
                original_system_prompt = final_messages[0].get("content", "")
                final_messages[0]["content"] = f"{original_system_prompt}\n\n{personality_prompt}"
            else:
                # 如果沒有系統提示，插入一個
                final_messages.insert(0, {"role": "system", "content": personality_prompt})

        # 如果設定不使用流式響應且開啟緩存，嘗試從緩存獲取
        cache_key_data = None
        if not stream and use_cache and self.cache_service:
            cache_key_data = {
                "model_id": model_id,
                "messages": final_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "humanize": humanize # 將humanize也加入緩存鍵
            }
            cached_response = self.cache_service.get("model_call", cache_key_data)
            if cached_response:
                logger.info(f"Cache hit for model {model_id}")
                return json.loads(cached_response)

        # 獲取模型信息以設置API密鑰
        model_info = self.model_service.get_model_by_id(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found in the system")

        provider = model_info.provider.lower()
        api_key = None
        if provider == "openrouter":
            api_key = self.api_key_manager.get_openrouter_key()
        else:
            api_key = self.api_key_manager.get_provider_key(provider)

        if not api_key:
            raise ValueError(f"No API key available for provider {provider}")

        # 獲取 LiteLLM 需要的模型標識符
        litellm_model_id = litellm.model_alias_map.get(model_id, model_id)

        # 設置 LiteLLM 的 API Key
        api_base = None
        custom_llm_provider = None

        if provider == "google":
             # Google Gemini 可能需要特殊處理或通過 Vertex AI
             pass # LiteLLM 通常能自動處理
        elif provider == "openai":
            pass # LiteLLM 默認處理
        elif provider == "nvidia":
            # LiteLLM 可能需要指定 API base
            api_base = "https://integrate.api.nvidia.com/v1"
        elif provider == "grok":
             # Grok 可能需要特定配置
             pass
        elif provider == "mistral":
             pass
        # 注意：OpenRouter 的 API Key 是在調用時傳遞

        # 調用模型，增加超時和重試邏輯
        response = None
        last_exception = None
        for attempt in range(litellm.num_retries + 1):
            try:
                start_time = time.time()
                logger.debug(f"Attempt {attempt+1} calling model {litellm_model_id}")

                # 準備參數，特別是OpenRouter的key
                call_kwargs = {
                    "model": litellm_model_id,
                    "messages": final_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": stream,
                    "timeout": litellm.request_timeout,
                }
                if provider == "openrouter":
                    call_kwargs["api_key"] = api_key
                elif api_base:
                     call_kwargs["api_base"] = api_base
                # 其他提供商的密鑰通常由LiteLLM自動從環境變量讀取

                response = await acompletion(**call_kwargs)

                # 檢查響應是否有效
                if not response:
                    raise ValueError("Received empty response from the model")

                # 記錄成功調用的延遲
                end_time = time.time()
                latency = end_time - start_time
                logger.info(f"Model {model_id} call succeeded in {latency:.2f}s on attempt {attempt+1}")

                # 如果未使用流式響應且開啟緩存，將結果存入緩存
                if not stream and use_cache and self.cache_service and cache_key_data:
                    self.cache_service.set(
                        "model_call",
                        cache_key_data,
                        json.dumps(response.dict()), # 使用 .dict() 轉換 Pydantic 模型
                        expire_seconds=3600
                    )
                return response.dict() # 返回字典

            except asyncio.TimeoutError:
                last_exception = asyncio.TimeoutError(f"Model {model_id} request timed out")
                logger.warning(f"Timeout on attempt {attempt+1} for model {model_id}")
            except litellm.exceptions.RateLimitError as e:
                 last_exception = e
                 logger.warning(f"Rate limit error on attempt {attempt+1} for model {model_id}: {e}")
                 # 如果是OpenRouter密鑰限流，嘗試立即更換密鑰並重試
                 if provider == "openrouter":
                     logger.info("Attempting to rotate OpenRouter key due to rate limit.")
                     api_key = self.api_key_manager.get_openrouter_key() # 獲取下一個密鑰
                     if attempt < litellm.num_retries: continue # 立即重試
                 # 其他提供商或最後一次嘗試，則按規則等待
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt+1} failed for model {model_id}: {str(e)}")

            # 如果不是最後一次嘗試，則等待後重試
            if attempt < litellm.num_retries:
                await asyncio.sleep(1.5 * (attempt + 1)) # 指數退避 + 隨機抖動
            else:
                 logger.error(f"All attempts failed for model {model_id}. Last exception: {last_exception}")
                 raise last_exception

        # 理論上不應該執行到這裡
        raise ValueError(f"Model call failed unexpectedly for {model_id}")

    async def call_model(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
        use_cache: bool = True,
        humanize: bool = True,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """公開的調用單個模型的方法"""
        return await self._call_model_with_retry(
            model_id, messages, temperature, max_tokens, stream, use_cache, humanize, user_id
        )

    async def call_multiple_models(
        self,
        model_ids: List[str],
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        use_cache: bool = True,
        humanize: bool = True,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """並行調用多個模型，改進并發處理和錯誤報告"""
        results = {}

        # 創建並行任務
        async def call_model_safe(model_id):
            try:
                model_response = await self.call_model(
                    model_id=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False, # 多模型調用通常不使用流式
                    use_cache=use_cache,
                    humanize=humanize,
                    user_id=user_id
                )
                return model_id, model_response, None
            except Exception as e:
                # 更詳細的錯誤記錄
                logger.error(f"Error calling model {model_id} in parallel: {type(e).__name__} - {str(e)}")
                return model_id, None, str(e)

        # 使用信號量限制並發數量
        max_concurrent = min(len(model_ids), 5) # 最多5個並發請求
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_call(model_id):
            async with semaphore:
                return await call_model_safe(model_id)

        # 並行執行所有模型調用
        tasks = [limited_call(model_id) for model_id in model_ids]
        responses = await asyncio.gather(*tasks)

        # 處理結果
        for model_id, response, error in responses:
            if error:
                results[model_id] = {"error": error}
            else:
                results[model_id] = response # 直接存儲字典響應

        return results

    async def debate_between_models(
        self,
        model_ids: List[str],
        topic: str,
        debate_format: DebateFormat, # 使用定義好的格式模型
        user_id: Optional[str] = None,
        humanize: bool = True
    ) -> List[Dict[str, Any]]:
        """讓模型進行辯論，允許設定辯論規則和評判標準"""
        if len(model_ids) < 2:
            raise ValueError("At least two models are required for a debate")

        # 創建辯論提示
        if humanize:
            personality_prompt = self.personality_service.get_personality_prompt(user_id)
            system_prompt = f"{self._create_debate_system_prompt(topic, debate_format)}\n\n{personality_prompt}"
        else:
            system_prompt = self._create_debate_system_prompt(topic, debate_format)

        # 初始化辯論歷史
        debate_history = []
        debate_messages = [{"role": "system", "content": system_prompt}]

        # 進行多輪辯論
        for round_num in range(debate_format.rounds):
            round_title = self._get_round_title(round_num, debate_format.rounds)
            round_responses = {}
            logger.info(f"Starting debate {round_title}")

            # 為每個模型創建本輪提示
            tasks = []
            for model_id in model_ids:
                 tasks.append(self._prepare_and_call_debate_model(round_num, model_id, debate_messages, topic, debate_history, debate_format, user_id))

            # 並行獲取所有模型的回應
            round_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 處理本輪結果
            temp_debate_messages = [] # 臨時存儲本輪的消息，避免修改主列表
            for i, result in enumerate(round_results):
                model_id = model_ids[i]
                model_info = self.model_service.get_model_by_id(model_id)
                model_name = model_info.name if model_info else model_id

                if isinstance(result, Exception):
                    logger.error(f"Error in debate round {round_num} with model {model_id}: {str(result)}")
                    error_msg = f"[無法回應，可能是模型暫時不可用或請求超過限制。 Error: {type(result).__name__}]"
                    round_responses[model_id] = error_msg
                    temp_debate_messages.append({"role": "assistant", "content": f"[{model_name}]: {error_msg}"})
                    # 記錄錯誤到性能分析
                    if self.analytics_service:
                         # ... (省略性能記錄代碼)
                         pass
                else:
                    response_content = result
                    round_responses[model_id] = response_content
                    temp_debate_messages.append({"role": "assistant", "content": f"[{model_name}]: {response_content}"})
                    # 記錄成功到性能分析
                    if self.analytics_service:
                        # ... (省略性能記錄代碼)
                        pass

            # 將本輪所有消息加入主辯論歷史
            debate_messages.extend(temp_debate_messages)

            # 如果設置了評判標準，進行評判 (可異步執行)
            round_evaluation = None
            if debate_format.judge_criteria and round_num > 0:
                try:
                    round_evaluation = await self._evaluate_debate_round(
                        topic,
                        round_responses,
                        debate_format.judge_criteria,
                        round_num
                    )
                except Exception as e:
                    logger.error(f"Error evaluating debate round {round_num}: {str(e)}")

            # 將該輪記錄添加到辯論歷史
            debate_history.append({
                "round": round_num + 1,
                "title": round_title,
                "responses": round_responses,
                "evaluation": round_evaluation
            })

            # 如果有評判結果，添加到對話中（可選）
            if round_evaluation:
                debate_messages.append({"role": "system", "content": f"[Round {round_num+1} Evaluation]: {round_evaluation}"})

            # 短暫延遲以避免API限制
            await asyncio.sleep(0.5)

        # 生成最終評判結果
        final_evaluation = None
        try:
            final_evaluation = await self._generate_final_debate_evaluation(
                topic,
                debate_history,
                debate_format.judge_criteria
            )
        except Exception as e:
            logger.error(f"Error generating final debate evaluation: {str(e)}")

        # 添加最終評判結果
        if final_evaluation:
            debate_history.append({
                "round": "final",
                "title": "辯論總結評判",
                "evaluation": final_evaluation
            })

        return debate_history

    async def _prepare_and_call_debate_model(
        self,
        round_num: int,
        model_id: str,
        debate_messages: List[Dict[str, Any]],
        topic: str,
        debate_history: List[Dict[str, Any]],
        debate_format: DebateFormat,
        user_id: Optional[str] = None
    ) -> str:
        """準備並調用單個模型進行辯論"""
        model_info = self.model_service.get_model_by_id(model_id)
        model_name = model_info.name if model_info else model_id

        # 準備提示信息
        model_messages = debate_messages.copy()
        instruction = self._create_round_instruction(
            round_num,
            model_name,
            topic,
            debate_history,
            model_id,
            debate_format
        )
        model_messages.append({"role": "user", "content": instruction})

        start_time = time.time()
        response_dict = await self.call_model(
            model_id=model_id,
            messages=model_messages,
            temperature=0.7,
            max_tokens=debate_format.time_per_round,
            use_cache=False, # 辯論不使用緩存
            humanize=False, # 人性化在系統提示中處理
            user_id=user_id
        )
        end_time = time.time()

        # 提取內容並記錄性能
        response_content = response_dict.get("choices", [{}])[0].get("message", {}).get("content", "")
        if self.analytics_service:
            # 記錄性能數據...
            pass

        return response_content

    # _create_debate_system_prompt, _get_round_title, _create_round_instruction, _evaluate_debate_round, _generate_final_debate_evaluation 方法實現省略...

    # --- 混合策略實現 --- #
    async def blend_model_responses(
        self,
        model_ids: List[str],
        messages: List[Dict[str, Any]],
        blend_method: str = "weighted",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        weights: Optional[Dict[str, float]] = None,
        humanize: bool = True,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """混合多個模型的回應 - 增強版本，支持更多策略"""
        if len(model_ids) < 2:
            raise ValueError("At least two models are required for blending")

        # 獲取所有模型回應
        model_responses = await self.call_multiple_models(
            model_ids=model_ids,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=True, # 混合前的單獨調用可以使用緩存
            humanize=humanize,
            user_id=user_id
        )

        # 提取有效回應
        valid_responses = {}
        for model_id, response in model_responses.items():
            if "error" in response:
                logger.warning(f"Skipping model {model_id} in blending due to error: {response['error']}")
                continue

            model_info = self.model_service.get_model_by_id(model_id)
            model_name = model_info.name if model_info else model_id

            try:
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content and len(content.strip()) > 0:
                    valid_responses[model_name] = {
                        "content": content,
                        "priority": model_info.priority if model_info else 0,
                        "model_id": model_id # 保存原始ID
                    }
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"Error extracting response from {model_id}: {str(e)}")

        # 檢查有效回應數量
        if len(valid_responses) < 2:
            if len(valid_responses) == 1:
                model_name, data = next(iter(valid_responses.items()))
                logger.warning(f"Only one valid response from {model_name}, returning without blending")
                return {
                    "content": data["content"],
                    "sources": [model_name],
                    "blend_method": "single_response"
                }
            else:
                raise ValueError("No valid responses to blend")

        # 根據選擇的方法混合回應
        try:
            if blend_method == "weighted":
                result = await self._weighted_blend(valid_responses, messages, weights, humanize, user_id)
            elif blend_method == "ensemble":
                result = await self._ensemble_blend(valid_responses, messages, humanize, user_id)
            elif blend_method == "chain":
                result = await self._chain_blend(valid_responses, messages, humanize, user_id)
            elif blend_method == "specialized":
                result = await self._specialized_blend(valid_responses, messages, humanize, user_id)
            else:
                # 默認使用加權混合
                logger.warning(f"Unknown blend method '{blend_method}', falling back to 'weighted'")
                result = await self._weighted_blend(valid_responses, messages, weights, humanize, user_id)
            return result
        except Exception as e:
            logger.error(f"Error during blending process (method: {blend_method}): {str(e)}")
            return self._fallback_blend(valid_responses)

    # --- 各種混合策略的內部實現 --- #
    async def _call_blender_model(
        self,
        prompt: str,
        system_message: str,
        humanize: bool,
        user_id: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        preferred_blender_model_id: Optional[str] = None
    ) -> str:
        """內部函數，用於調用執行混合的模型"""
        # 選擇混合模型
        blender_model_id = preferred_blender_model_id
        if not blender_model_id:
            # 默認選擇最高優先級的模型
            top_models = self.model_service.get_top_models(limit=1)
            if top_models:
                blender_model_id = top_models[0].id
            else:
                 # 如果沒有模型，使用第一個可用模型
                 blender_model_id = list(self.model_service.models.keys())[0]

        # 準備消息
        blend_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        # 調用模型
        blend_response_dict = await self.call_model(
            model_id=blender_model_id,
            messages=blend_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=False, # 混合過程不使用緩存
            humanize=humanize, # 應用人性化到混合器
            user_id=user_id
        )

        return blend_response_dict.get("choices", [{}])[0].get("message", {}).get("content", "")

    async def _weighted_blend(self, valid_responses, messages, weights, humanize, user_id):
        """基於模型權重的混合策略"""
        user_query = messages[-1]["content"] if messages else ""
        if not weights:
            weights = {name: data["priority"] for name, data in valid_responses.items()}
        # ... (提示構建邏輯) ...
        prompt = "..."
        system_message = "您是一位專業的內容綜合專家..."

        blended_content = await self._call_blender_model(prompt, system_message, humanize, user_id, temperature=0.4)

        return {
            "content": blended_content,
            "sources": list(valid_responses.keys()),
            "blend_method": "weighted",
            # "blender_model": blender_model_name # 需要從 _call_blender_model 返回
        }

    async def _ensemble_blend(self, valid_responses, messages, humanize, user_id):
        """使用集成學習方法混合模型回應"""
        user_query = messages[-1]["content"] if messages else ""
        # ... (句子分割、評分邏輯) ...
        prompt = "..."
        system_message = "您是一位專業的內容集成專家..."

        blended_content = await self._call_blender_model(prompt, system_message, humanize, user_id, temperature=0.3)

        return {
            "content": blended_content,
            "sources": list(valid_responses.keys()),
            "blend_method": "ensemble",
            # "blender_model": blender_model_name
        }

    async def _chain_blend(self, valid_responses, messages, humanize, user_id):
        """使用鏈式方法混合模型回應"""
        user_query = messages[-1]["content"] if messages else ""
        # ... (模型排序、鏈式調用邏輯) ...
        final_response_content = "..."
        chain_history = [] # 記錄鏈式步驟

        return {
            "content": final_response_content,
            "sources": [item["model"] for item in chain_history],
            "blend_method": "chain",
            "chain_history": chain_history
        }

    async def _specialized_blend(self, valid_responses, messages, humanize, user_id):
        """根據問題類型選擇專業模型回應，並增強混合"""
        user_query = messages[-1]["content"] if messages else ""
        # ... (問題分析、模型評分、選擇專家邏輯) ...
        prompt = "..."
        system_message = "您是一位專業的內容整合專家..."

        # 選擇評判者模型ID
        judge_model_id = "..."

        blended_content = await self._call_blender_model(prompt, system_message, humanize, user_id, temperature=0.3, preferred_blender_model_id=judge_model_id)

        return {
            "content": blended_content,
            "sources": [], # 記錄專家模型
            "judge": "", # 記錄評判模型
            "blend_method": "specialized",
            "question_types": {},
            "model_scores": {}
        }

    # 輔助函數：句子相似度計算
    def _sentence_similarity(self, s1, s2):
        """計算兩個句子的相似度（簡單版本）"""
        s1_words = set(re.findall(r'\w+', s1.lower()))
        s2_words = set(re.findall(r'\w+', s2.lower()))
        if not s1_words or not s2_words:
            return 0.0
        intersection = len(s1_words.intersection(s2_words))
        union = len(s1_words.union(s2_words))
        return intersection / union

    # 退化混合方案
    def _fallback_blend(self, valid_responses):
        """提供一個退化的混合方案，當高級混合失敗時使用"""
        combined_content = "由於複雜混合處理失敗，以下是各模型的原始回應:\n\n"
        for name, data in valid_responses.items():
            combined_content += f"## {name}:\n{data['content']}\n\n"
        return {
            "content": combined_content,
            "sources": list(valid_responses.keys()),
            "blend_method": "fallback_compilation",
            "error": "Advanced blending failed"
        }

    # --- 辯論輔助方法 --- #
    def _create_debate_system_prompt(self, topic: str, debate_format: DebateFormat) -> str:
        """創建辯論系統提示"""
        # ... (實現省略)
        return ""

    def _get_round_title(self, round_num: int, total_rounds: int) -> str:
        """獲取辯論輪次標題"""
        # ... (實現省略)
        return f"Round {round_num + 1}"

    def _create_round_instruction(
        self, round_num: int, model_name: str, topic: str,
        debate_history: List[Dict[str, Any]], model_id: str,
        debate_format: DebateFormat
    ) -> str:
        """創建辯論輪次指示"""
        # ... (實現省略)
        return ""

    async def _evaluate_debate_round(
        self, topic: str, round_responses: Dict[str, str],
        criteria: List[str], round_num: int
    ) -> Optional[str]:
        """評估辯論輪次"""
        # ... (實現省略)
        return None

    async def _generate_final_debate_evaluation(
        self, topic: str, debate_history: List[Dict[str, Any]],
        criteria: List[str]
    ) -> Optional[str]:
        """生成最終辯論評判"""
        # ... (實現省略)
        return None 