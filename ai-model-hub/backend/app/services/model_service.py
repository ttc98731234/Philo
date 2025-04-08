from typing import List, Dict, Any, Optional, Set
import logging
from pydantic import BaseModel
import asyncio
import time
import re
import litellm
from litellm.exceptions import APIConnectionError, AuthenticationError, RateLimitError, BadRequestError, ServiceUnavailableError

logger = logging.getLogger(__name__)

class ModelCapability(BaseModel):
    name: str
    description: str

class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str
    description: str = ""
    descriptions: Optional[Dict[str, str]] = None # 添加多語言描述
    capabilities: List[str] = []
    priority: int = 0
    available: bool = True
    max_tokens: int = 4096
    context_size: int = 8192
    cost_per_1k_tokens: float = 0.0
    response_time_estimate: float = 1.0  # 秒

class ModelService:
    def __init__(self):
        # 初始化模型能力定義
        self.capabilities = self._initialize_capabilities()

        # 初始化可用模型列表
        self.models = self._initialize_models()

        # 初始化模型可用性跟踪
        self.model_availability = {model_id: True for model_id in self.models.keys()}
        self.availability_check_time = time.time()
        # 添加檢查時間間隔（例如，5分鐘）
        self.check_interval = 300 # seconds

    def _initialize_capabilities(self) -> Dict[str, ModelCapability]:
        """初始化模型能力定義"""
        # 這裡可以添加更多能力或細化現有能力
        return {
            "text": ModelCapability(
                name="文本處理",
                description="基本的文本理解和生成能力"
            ),
            "reasoning": ModelCapability(
                name="推理能力",
                description="邏輯分析、問題解決和推理能力"
            ),
            "math": ModelCapability(
                name="數學能力",
                description="數學計算、解方程和數學推理能力"
            ),
            "coding": ModelCapability(
                name="編程能力",
                description="生成和理解程式碼的能力"
            ),
            "creative": ModelCapability(
                name="創意寫作",
                description="生成創意內容，如故事、詩歌等"
            ),
            "images": ModelCapability(
                name="圖像理解",
                description="理解和描述圖像內容的能力"
            ),
            "multilingual": ModelCapability(
                name="多語言支持",
                description="支持多種語言的處理能力"
            )
        }

    def _initialize_models(self) -> Dict[str, ModelInfo]:
        """初始化所有支持的模型，包含多語言描述"""
        models = {}

        # 定義所有支持的模型（按照優先級順序）
        model_configs = [
            {
                "id": "gemini-2.5-pro-exp-03-25",
                "name": "Gemini 2.5 Pro Exp",
                "provider": "google",
                "description": "Google's most advanced model with multi-modal capabilities and enhanced reasoning",
                "descriptions": {
                    "zh-TW": "Google 最先進的模型，具有多模態能力和增強的推理能力",
                    "zh-CN": "谷歌最先进的模型，具有多模态能力和增强的推理能力"
                },
                "capabilities": ["text", "images", "reasoning", "math", "coding", "multilingual"],
                "priority": 15,
                "max_tokens": 8192,
                "context_size": 32768,
                "cost_per_1k_tokens": 0.0007,
                "response_time_estimate": 1.2
            },
            {
                "id": "llama-4-M",
                "name": "Llama 4 M",
                "provider": "openrouter",
                "description": "Meta's advanced open-source LLM with strong reasoning and efficiency improvements",
                 "descriptions": {
                    "zh-TW": "Meta 的高級開源 LLM，具有強大的推理能力和效率改進",
                    "zh-CN": "Meta 的高级开源 LLM，具有强大的推理能力和效率改进"
                },
                "capabilities": ["text", "reasoning", "math", "coding"],
                "priority": 14,
                "max_tokens": 4096,
                "context_size": 16384,
                "cost_per_1k_tokens": 0.0004,
                "response_time_estimate": 1.4
            },
            {
                "id": "deepseek-r1",
                "name": "DeepSeek R1",
                "provider": "nvidia",
                "description": "DeepSeek's advanced reasoning model designed for complex problem-solving",
                 "descriptions": {
                    "zh-TW": "DeepSeek 的高級推理模型，專為解決複雜問題而設計",
                    "zh-CN": "DeepSeek 的高级推理模型，专为解决复杂问题而设计"
                },
                "capabilities": ["text", "reasoning", "math", "coding"],
                "priority": 13,
                "max_tokens": 4096,
                "context_size": 16384,
                "cost_per_1k_tokens": 0.0006,
                "response_time_estimate": 1.5
            },
            {
                "id": "deepseek-v3-0324",
                "name": "DeepSeek V3",
                "provider": "openrouter",
                "description": "DeepSeek's latest large language model with enhanced capabilities",
                 "descriptions": {
                    "zh-TW": "DeepSeek 最新的大型語言模型，具有增強的能力",
                    "zh-CN": "DeepSeek 最新的大型语言模型，具有增强的能力"
                },
                "capabilities": ["text", "coding", "reasoning", "math"],
                "priority": 12,
                "max_tokens": 4096,
                "context_size": 16384,
                "cost_per_1k_tokens": 0.0005,
                "response_time_estimate": 1.3
            },
            {
                "id": "gemini-2.0-pro-exp-02-05",
                "name": "Gemini 2.0 Pro Exp",
                "provider": "google",
                "description": "Google's Pro model with expanded capabilities in reasoning and instructions",
                 "descriptions": {
                    "zh-TW": "Google 的 Pro 模型，具有擴展的推理和指令能力",
                    "zh-CN": "谷歌的 Pro 模型，具有扩展的推理和指令能力"
                },
                "capabilities": ["text", "images", "reasoning"],
                "priority": 11,
                "max_tokens": 4096,
                "context_size": 16384,
                "cost_per_1k_tokens": 0.0005,
                "response_time_estimate": 1.0
            },
            {
                "id": "gemini-2.0-flash-thinking-exp-01-21",
                "name": "Gemini 2.0 Flash Thinking",
                "provider": "google",
                "description": "Fast and efficient Gemini model with enhanced reasoning",
                 "descriptions": {
                    "zh-TW": "快速高效的 Gemini 模型，具有增強的推理能力",
                    "zh-CN": "快速高效的 Gemini 模型，具有增强的推理能力"
                },
                "capabilities": ["text", "reasoning"],
                "priority": 10,
                "max_tokens": 2048,
                "context_size": 8192,
                "cost_per_1k_tokens": 0.0003,
                "response_time_estimate": 0.8
            },
            {
                "id": "grok2-1212",
                "name": "Grok 2",
                "provider": "grok",
                "description": "xAI's advanced conversational model with up-to-date knowledge and reasoning",
                 "descriptions": {
                    "zh-TW": "xAI 的高級對話模型，具有最新的知識和推理能力",
                    "zh-CN": "xAI 的高级对话模型，具有最新的知识和推理能力"
                },
                "capabilities": ["text", "reasoning", "coding", "math"],
                "priority": 9,
                "max_tokens": 4096,
                "context_size": 12288,
                "cost_per_1k_tokens": 0.0006,
                "response_time_estimate": 1.2
            },
            {
                "id": "perxparity",
                "name": "Perxparity",
                "provider": "openrouter",
                "description": "Balanced model with good reasoning capabilities for general text tasks",
                 "descriptions": {
                    "zh-TW": "平衡的模型，具有良好的推理能力，適用於一般文本任務",
                    "zh-CN": "平衡的模型，具有良好的推理能力，适用于一般文本任务"
                },
                "capabilities": ["text", "reasoning"],
                "priority": 8,
                "max_tokens": 2048,
                "context_size": 8192,
                "cost_per_1k_tokens": 0.0003,
                "response_time_estimate": 0.9
            },
            {
                "id": "gemini-2.0-flash-001",
                "name": "Gemini 2.0 Flash",
                "provider": "google",
                "description": "Efficient Gemini model for general text tasks with good performance",
                 "descriptions": {
                    "zh-TW": "高效的 Gemini 模型，適用於一般文本任務，性能良好",
                    "zh-CN": "高效的 Gemini 模型，适用于一般文本任务，性能良好"
                },
                "capabilities": ["text"],
                "priority": 7,
                "max_tokens": 2048,
                "context_size": 8192,
                "cost_per_1k_tokens": 0.0003,
                "response_time_estimate": 0.7
            },
            {
                "id": "gemma-3-27b-it",
                "name": "Gemma 3 27B IT",
                "provider": "nvidia",
                "description": "Google's Gemma 3 model with 27B parameters, instruction-tuned",
                 "descriptions": {
                    "zh-TW": "Google 的 Gemma 3 模型，具有 27B 參數，經過指令調整",
                    "zh-CN": "谷歌的 Gemma 3 模型，具有 27B 参数，经过指令调整"
                },
                "capabilities": ["text", "coding", "reasoning"],
                "priority": 6,
                "max_tokens": 4096,
                "context_size": 8192,
                "cost_per_1k_tokens": 0.0004,
                "response_time_estimate": 1.0
            },
            {
                "id": "deepseek-v3",
                "name": "DeepSeek V3",
                "provider": "openrouter",
                "description": "General DeepSeek language model with balanced capabilities",
                 "descriptions": {
                    "zh-TW": "通用的 DeepSeek 語言模型，具有平衡的能力",
                    "zh-CN": "通用的 DeepSeek 语言模型，具有平衡的能力"
                },
                "capabilities": ["text", "coding"],
                "priority": 5,
                "max_tokens": 4096,
                "context_size": 8192,
                "cost_per_1k_tokens": 0.0004,
                "response_time_estimate": 1.1
            },
            {
                "id": "qwq-32b",
                "name": "QwQ 32B",
                "provider": "openrouter",
                "description": "Large parameter model with creative capabilities and general knowledge",
                 "descriptions": {
                    "zh-TW": "大型參數模型，具有創意能力和通用知識，特別擅長中文",
                    "zh-CN": "大型参数模型，具有创意能力和通用知识，特别擅长中文"
                },
                "capabilities": ["text", "creative", "coding", "multilingual"],
                "priority": 4,
                "max_tokens": 4096,
                "context_size": 8192,
                "cost_per_1k_tokens": 0.0005,
                "response_time_estimate": 1.3
            },
            {
                "id": "gemini-2.0-flash-lite",
                "name": "Gemini 2.0 Flash Lite",
                "provider": "google",
                "description": "Lightweight Gemini model optimized for speed and efficiency",
                 "descriptions": {
                    "zh-TW": "輕量級 Gemini 模型，為速度和效率進行了優化",
                    "zh-CN": "轻量级 Gemini 模型，为速度和效率进行了优化"
                },
                "capabilities": ["text"],
                "priority": 3,
                "max_tokens": 1024,
                "context_size": 4096,
                "cost_per_1k_tokens": 0.0002,
                "response_time_estimate": 0.6
            },
            {
                "id": "gemini-1.5-pro-002",
                "name": "Gemini 1.5 Pro",
                "provider": "google",
                "description": "Previous generation Gemini Pro model with multimodal capabilities",
                 "descriptions": {
                    "zh-TW": "上一代 Gemini Pro 模型，具有多模態能力",
                    "zh-CN": "上一代 Gemini Pro 模型，具有多模态能力"
                },
                "capabilities": ["text", "images"],
                "priority": 2,
                "max_tokens": 2048,
                "context_size": 8192,
                "cost_per_1k_tokens": 0.0004,
                "response_time_estimate": 0.9
            },
            {
                "id": "llama-3.3-nemotron-super-49b-v1",
                "name": "Llama 3.3 Nemotron Super 49B",
                "provider": "nvidia",
                "description": "Large parameter Llama model with comprehensive capabilities",
                 "descriptions": {
                    "zh-TW": "大型參數 Llama 模型，具有全面的能力",
                    "zh-CN": "大型参数 Llama 模型，具有全面的能力"
                },
                "capabilities": ["text", "coding", "reasoning"],
                "priority": 1,
                "max_tokens": 4096,
                "context_size": 8192,
                "cost_per_1k_tokens": 0.0005,
                "response_time_estimate": 1.4
            }
        ]

        for config in model_configs:
            # 驗證能力是否已定義
            valid_capabilities = [cap for cap in config.get("capabilities", []) if cap in self.capabilities]
            if len(valid_capabilities) != len(config.get("capabilities", [])):
                invalid_caps = set(config.get("capabilities", [])) - set(valid_capabilities)
                logger.warning(f"模型 '{config['id']}' 包含未定義的能力: {invalid_caps}. 這些能力將被忽略。")
            
            # 檢查描述字段
            if "description" not in config or not config["description"]:
                 logger.warning(f"模型 '{config['id']}' 缺少 'description' 字段。")
                 config["description"] = "" # 提供默認值
            
            # 確保 descriptions 是字典或 None
            if "descriptions" in config and not isinstance(config.get("descriptions"), (dict, type(None))):
                 logger.warning(f"模型 '{config['id']}' 的 'descriptions' 字段不是有效的字典。將設置為 None。")
                 config["descriptions"] = None

            models[config["id"]] = ModelInfo(**config, capabilities=valid_capabilities)

        # 按優先級排序模型字典 (雖然字典本身是無序的，但在需要排序列表時這很有用)
        # 注意：Python 3.7+ 字典保持插入順序，但這裡的模型是按 config 順序插入的，
        # 我們希望的可能是按 priority 排序。如果需要按 priority 獲取，get_top_models 做了這個。
        # sorted_models = dict(sorted(models.items(), key=lambda item: item[1].priority, reverse=True))
        # return sorted_models
        # 保持插入順序即可，因為 get_top_models 會處理排序
        return models

    async def update_model_availability(self, model_id: str, available: bool):
        """異步更新指定模型的可用狀態"""
        if model_id in self.models:
            # 同時更新 ModelInfo 對象和可用性字典
            self.models[model_id].available = available
            self.model_availability[model_id] = available
            logger.info(f"模型 '{model_id}' 的可用狀態已更新為: {available}")
        else:
            logger.warning(f"嘗試更新不存在的模型 '{model_id}' 的可用性。")

    async def _check_single_model(self, model_id: str) -> bool:
        """異步檢查單個模型的可用性"""
        model_info = self.models.get(model_id)
        if not model_info:
            return False # 模型不存在

        # 使用 LiteLLM 需要的格式，例如 "openai/gpt-4" 或 "gemini/gemini-pro"
        # 注意：map_model_to_provider_format 假設模型 ID 就是 LiteLLM 需要的格式
        # 如果不是，需要調整 map_model_to_provider_format 的邏輯
        litellm_model_name = self.map_model_to_provider_format(model_id, model_info.provider)
        if not litellm_model_name:
             logger.error(f"無法映射模型 ID '{model_id}' 到提供商 '{model_info.provider}' 的格式。跳過檢查。")
             return False # 或者根據情況返回之前的狀態？暫定為不可用

        try:
            # 使用一個非常簡單且便宜的請求來測試連通性
            logger.debug(f"正在檢查模型 '{litellm_model_name}' ({model_id}) 的可用性...")
            await litellm.completion(
                model=litellm_model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                temperature=0,
                timeout=10 # 設置較短的超時時間
            )
            logger.info(f"模型 '{litellm_model_name}' ({model_id}) 可用。")
            return True
        except (APIConnectionError, AuthenticationError, RateLimitError, BadRequestError, ServiceUnavailableError) as e:
            logger.warning(f"檢查模型 '{litellm_model_name}' ({model_id}) 時出錯: {type(e).__name__} - {e}")
            return False
        except Exception as e:
            logger.error(f"檢查模型 '{litellm_model_name}' ({model_id}) 時發生意外錯誤: {e}", exc_info=True)
            return False

    async def check_models_availability(self, force: bool = False):
        """異步檢查所有模型的可用性，除非最近剛檢查過。"""
        current_time = time.time()
        if not force and (current_time - self.availability_check_time < self.check_interval):
            logger.debug(f"最近已檢查模型可用性，跳過。下次檢查在 {self.availability_check_time + self.check_interval - current_time:.1f} 秒後。")
            return

        logger.info("開始執行模型可用性異步檢查...")
        tasks = []
        model_ids_to_check = list(self.models.keys()) # 獲取當前所有模型 ID

        for model_id in model_ids_to_check:
            # 為每個模型創建一個檢查任務
            tasks.append(self._check_single_model(model_id))

        # 並行執行所有檢查任務
        results = await asyncio.gather(*tasks, return_exceptions=True) # return_exceptions 以處理任務中的錯誤

        new_availability = {}
        for model_id, result in zip(model_ids_to_check, results):
            if isinstance(result, Exception):
                # 如果 gather 返回的是異常，說明 _check_single_model 內部拋出了未捕獲的異常
                logger.error(f"檢查模型 '{model_id}' 的任務失敗: {result}", exc_info=result)
                is_available = False # 將任務失敗視為不可用
            else:
                # result 是 _check_single_model 的返回值 (True/False)
                is_available = result

            # 更新可用性字典
            new_availability[model_id] = is_available
            # 同步更新 ModelInfo 對象中的狀態 (可選，取決於您希望如何使用該狀態)
            if model_id in self.models:
                self.models[model_id].available = is_available

        self.model_availability = new_availability
        self.availability_check_time = time.time()
        logger.info(f"模型可用性檢查完成。當前可用模型數量: {sum(new_availability.values())}/{len(new_availability)}")
        logger.debug(f"詳細可用性狀態: {new_availability}")

    def get_all_models(self) -> List[ModelInfo]:
        """獲取所有模型的列表，包含當前可用性狀態"""
        all_models = list(self.models.values())
        # 更新每個模型的當前可用性狀態
        for model in all_models:
            model.available = self.model_availability.get(model.id, True)
        return all_models

    def get_all_capabilities(self) -> List[ModelCapability]:
        """取得所有模型能力的列表"""
        return list(self.capabilities.values())

    def get_model_by_id(self, model_id: str) -> Optional[ModelInfo]:
        """根據ID取得特定模型，包括可用性狀態"""
        model = self.models.get(model_id)
        if model:
            model.available = self.model_availability.get(model.id, True)
        return model

    def get_top_models(self, limit: int = 5) -> List[ModelInfo]:
        """根據優先級取得頂級模型，僅返回可用模型"""
        available_models = [m for m in self.get_all_models() if m.available]
        sorted_models = sorted(
            available_models,
            key=lambda x: x.priority,
            reverse=True
        )
        return sorted_models[:limit]

    def get_models_by_capability(self, capability: str) -> List[ModelInfo]:
        """根據能力篩選可用模型"""
        if capability not in self.capabilities:
            logger.warning(f"Unknown capability: {capability}")
            return []

        return [
            model for model in self.get_all_models()
            if model.available and capability in model.capabilities
        ]

    def get_models_by_task(self, task_description: str) -> List[ModelInfo]:
        """根據任務描述推薦最合適的模型列表"""
        # 1. (可選) 使用 NLP 模型分析任務描述，提取關鍵詞或意圖
        #    這裡簡化處理：直接使用正則表達式查找能力關鍵詞
        extracted_capabilities: Set[str] = set()
        task_lower = task_description.lower()

        # 簡單的關鍵詞映射 (可以擴展或使用更複雜的邏輯)
        keyword_to_capability = {
            '圖像': 'images', '圖片': 'images', '視覺': 'images', '看': 'images', '畫': 'images',
            '程式': 'coding', '代碼': 'coding', '編寫': 'coding', '開發': 'coding', 'debug': 'coding',
            '數學': 'math', '計算': 'math', '方程式': 'math', '算術': 'math',
            '推理': 'reasoning', '邏輯': 'reasoning', '分析': 'reasoning', '解決問題': 'reasoning',
            '創意': 'creative', '寫作': 'creative', '故事': 'creative', '詩': 'creative', '生成文本': 'text',
            '翻譯': 'multilingual', '多語言': 'multilingual', '語言': 'multilingual', # '語言' 可能太寬泛
            '文本': 'text', '摘要': 'text', '總結': 'text', '理解': 'text', # '理解' 可能太寬泛
        }

        # 改進關鍵詞匹配邏輯，避免部分匹配 (例如 'mathematics' 匹配 'math')
        # 使用正則表達式確保是獨立的詞
        for keyword, cap in keyword_to_capability.items():
            # 使用 \b 來匹配單詞邊界
            if re.search(r'\b' + re.escape(keyword) + r'\b', task_lower):
                extracted_capabilities.add(cap)

        # 如果沒有提取到特定能力，默認為 'text'
        if not extracted_capabilities:
            extracted_capabilities.add('text')
            logger.debug(f"未從任務 '{task_description}' 中提取到特定能力，默認為 'text'")
        else:
             logger.debug(f"從任務 '{task_description}' 中提取到能力: {extracted_capabilities}")

        # 2. 查找具備所有提取出的能力的模型
        suitable_models: List[ModelInfo] = []
        available_models = self.get_available_models() # 只在可用模型中查找

        for model in available_models:
            if extracted_capabilities.issubset(set(model.capabilities)):
                suitable_models.append(model)

        # 3. 按優先級排序
        suitable_models.sort(key=lambda m: m.priority, reverse=True)

        logger.info(f"為任務 '{task_description}' (需求能力: {extracted_capabilities}) 找到 {len(suitable_models)} 個合適的模型。")
        return suitable_models

    def get_available_models(self) -> List[ModelInfo]:
        """獲取當前標記為可用的模型列表"""
        # 使用 self.model_availability 字典來過濾
        return [model for model_id, model in self.models.items() if self.model_availability.get(model_id, False)]

    def map_model_to_provider_format(self, model_id: str, provider: str) -> Optional[str]:
        """將內部模型 ID 映射到特定提供商（如 LiteLLM）所需的格式。
           返回 None 表示無法映射或不支持。"""
        
        # 基礎假設：許多提供商的 LiteLLM 格式是 provider/model_name
        # 但需要處理特殊情況或不同的命名規則
        
        model_info = self.models.get(model_id)
        if not model_info:
            logger.warning(f"嘗試映射未知的模型 ID: {model_id}")
            return None

        # 檢查提供商是否匹配
        if model_info.provider.lower() != provider.lower():
             logger.warning(f"提供的提供商 '{provider}' 與模型 '{model_id}' 的記錄提供商 '{model_info.provider}' 不匹配。")
             # 根據策略，可以返回 None 或嘗試使用記錄的提供商
             # 這裡我們假設調用者知道他們想要的提供商，但最好還是基於模型自己的提供商
             provider = model_info.provider.lower() # 修正為使用模型記錄的提供商


        # === LiteLLM 格式映射規則 ===
        # Google Gemini: 通常是 "gemini/model-name" 或直接 "gemini-pro", "gemini-flash" 等
        # OpenAI: 通常是 "openai/model-name" 或直接 "gpt-4", "gpt-3.5-turbo"
        # Anthropic: 通常是 "anthropic/model-name"
        # Grok: 需要確認 LiteLLM 是否支持以及格式，可能是 "groq/model-name" 或需要 API key 指定
        # Nvidia: 可能需要特定的 API 端點或格式 "nvidia/model-name"
        # OpenRouter: 通常是 "openrouter/provider/model-name" 或直接 "openrouter/model-id"
        
        # 這裡的 model_id 已經是類似 "gemini-2.5-pro-exp-03-25" 的形式
        # 我們需要根據 provider 加上前綴（如果需要）

        provider_lower = provider.lower()

        # 針對 OpenRouter，其 model_id 通常已經包含了提供商信息（雖然我們的定義裡分開了）
        # LiteLLM 使用 openrouter/<model_id>
        if provider_lower == "openrouter":
            # 假設我們的 model_id 就是 OpenRouter 頁面上的模型 ID
            # 例如 'deepseek-v3-0324' -> 'openrouter/deepseek/deepseek-chat' 或 'openrouter/deepseek-v3-0324'？
            # 需要查閱 LiteLLM 文檔確認 OpenRouter 的確切格式。
            # 假設格式是 'openrouter/<我們的 model_id>'
             #return f"openrouter/{model_id}" 
             # OpenRouter 的模型 ID 通常可以直接用，不需要前綴
             # 例如：litellm.completion(model="openrouter/google/gemini-pro", ...)
             # 或者 litellm.completion(model="openrouter/anthropic/claude-3-opus", ...)
             # 我們的 model_id 似乎不是 OpenRouter 的原生 ID，而是我們自己定義的。
             # 這裡需要一個更明確的映射規則。
             # 暫時返回 model_id 本身，假設 LiteLLM 配置了 OpenRouter 的 base_url 和 api_key
             # 並且可以直接傳遞 'deepseek-v3-0324' 等 ID。這需要驗證。
             # 查閱 LiteLLM 文檔，OpenRouter 模型字符串通常是 'openrouter/<provider_shortname>/<model_name>'
             # 例如 'openrouter/google/gemini-pro'
             # 我們需要將我們的 'gemini-2.0-pro-exp-02-05' 轉換。這很複雜。
             # 簡化：假設我們的 model_id 在 LiteLLM 中可以直接用於 OpenRouter。
             # 如果不行，這裡需要大量 case-by-case 映射。
             # 更好的方法可能是在 ModelInfo 中存儲 litellm_id。
             logger.warning(f"OpenRouter 映射可能不準確，直接返回 model_id: {model_id}")
             return model_id # 假設 model_id 是 LiteLLM 可識別的 OpenRouter 模型 ID

        # 針對 Google Gemini
        elif provider_lower == "google":
             # LiteLLM 接受 'gemini/model-name' 或直接 'gemini-pro'
             # 我們的 ID 類似 'gemini-2.5-pro-exp-03-25'
             # 假設 LiteLLM 可以直接處理 'gemini/gemini-2.5-pro-exp-03-25'
             return f"gemini/{model_id}" # 返回 "gemini/gemini-..." 格式

        # 針對 Nvidia (NIM)
        elif provider_lower == "nvidia":
             # LiteLLM 可能需要 'nvidia/model-name'
             # 我們的 ID 類似 'deepseek-r1', 'gemma-3-27b-it'
             # 假設格式是 'nvidia/<model_id>'
             # return f"nvidia/{model_id}"
             # 查閱 LiteLLM 文檔，Nvidia NIM 需要設置 base_url，模型名稱可能就是 ID 本身
             logger.warning(f"Nvidia NIM 映射可能不準確，直接返回 model_id: {model_id}")
             return model_id # 假設 model_id 是 LiteLLM 在設置了 base_url 後可識別的

        # 針對 Grok
        elif provider_lower == "grok":
             # LiteLLM 的 Groq 提供商使用 'groq/model-name'
             # 我們的 ID 是 'grok2-1212'，模型名可能是 'gemma-7b-it' 或 'mixtral-8x7b-32768'
             # 這裡的映射不明確。
             # 假設我們的 model_id 'grok2-1212' 對應 Groq 上的某個模型，例如 'llama3-70b-8192'
             # 這需要手動映射。
             # if model_id == 'grok2-1212': return 'groq/llama3-70b-8192' # 示例
             logger.warning(f"Grok 映射不明確，直接返回 model_id: {model_id}")
             return model_id # 返回原始 ID，寄望於 LiteLLM 配置

        # 其他提供商 (需要根據 LiteLLM 文檔添加規則)
        # 例如 OpenAI:
        # elif provider_lower == "openai":
        #    return f"openai/{model_id}" # 或者直接 model_id 如果是 gpt-4 等

        # 默認回退：嘗試 provider/model_id 格式或直接 model_id
        else:
            logger.warning(f"提供商 '{provider}' 的映射規則未知，嘗試返回 model_id: {model_id}")
            # return f"{provider_lower}/{model_id}" # 這種格式不一定對
            return model_id # 更安全的做法是返回原始 ID，依賴 LiteLLM 配置

    def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        """估算使用特定模型處理給定數量的輸入和輸出 token 的成本。"""
        model = self.get_model_by_id(model_id)
        if not model:
            return 0.0

        # 使用模型的成本係數
        cost = model.cost_per_1k_tokens * (input_tokens + output_tokens) / 1000
        return round(cost, 6)

    def estimate_response_time(self, model_id: str, input_tokens: int, expected_output_tokens: int) -> float:
        """估算模型回應時間（秒）"""
        model = self.get_model_by_id(model_id)
        if not model:
            logger.warning(f"嘗試估算未知模型 '{model_id}' 的回應時間，返回默認值 3.0 秒。")
            return 3.0  # 默認估計

        # 基礎時間 + 基於輸入/輸出token的額外時間
        base_time = model.response_time_estimate
        # token 處理時間因子 (可以根據經驗調整，這裡假設每 1k token 增加 0.5 秒)
        # 注意：這是一個非常簡化的模型，實際時間受多種因素影響
        token_time = (input_tokens + expected_output_tokens) / 1000 * 0.5

        estimated_time = base_time + token_time
        logger.debug(f"模型 '{model_id}' 回應時間估算：基礎 {base_time:.1f}s + Token ({input_tokens}+{expected_output_tokens}) {token_time:.1f}s = {estimated_time:.1f}s")
        return round(estimated_time, 1) 