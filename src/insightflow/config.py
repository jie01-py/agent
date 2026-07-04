"""InsightFlow 配置管理。"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """LLM 服务配置。"""

    provider: str = "dashscope"
    model: str = "qwen-max"
    temperature: float = 0.3
    api_key: str = ""
    base_url: str = ""

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not self.base_url:
            self.base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


@dataclass
class PipelineConfig:
    """流程执行配置。"""

    output_dir: str = "output"
    max_iterations: int = 2
    human_review: bool = True
    verbose: bool = True
    chart_format: str = "png"
    chart_dpi: int = 150

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class InsightFlowConfig:
    """InsightFlow 系统顶层配置。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def load_config() -> InsightFlowConfig:
    """从环境变量加载配置。

    读取 DASHSCOPE_API_KEY 和 DASHSCOPE_MODEL，来源为 .env 文件或环境变量。
    """
    load_dotenv()

    return InsightFlowConfig(
        llm=LLMConfig(
            provider="dashscope",
            model=os.getenv("DASHSCOPE_MODEL", "qwen-max"),
            temperature=0.3,
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ),
        pipeline=PipelineConfig(
            output_dir=os.getenv("OUTPUT_DIR", "output"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "2")),
            human_review=os.getenv("HUMAN_REVIEW", "true").lower() == "true",
            verbose=os.getenv("VERBOSE", "true").lower() == "true",
        ),
    )


def get_chat_model(config: InsightFlowConfig | None = None):
    """根据配置创建并返回 LangChain ChatModel 实例。

    用于获取 LLM 聊天模型，所有 Agent 节点都通过此函数拿到模型实例。

    Args:
        config: 可选的配置覆盖。为 None 时自动从环境变量加载。

    Returns:
        LangChain ChatModel 实例（ChatOpenAI）。
    """
    from langchain_openai import ChatOpenAI

    if config is None:
        config = load_config()

    return ChatOpenAI(
        model=config.llm.model,
        temperature=config.llm.temperature,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
    )
