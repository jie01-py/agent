"""InsightFlow: 基于 LangGraph 和 MCP 的多 Agent 数据分析流程。

v2 新特性:
- MCP 协议深度: Resources、Prompts、RBAC 权限、DataFrameStore 缓存
- 会话级 DataFrame 上下文，支持版本控制和回滚
- 弹性 LLM 客户端，带指数退避和熔断机制
- 各 Agent 的 token 和成本追踪
- 结构化错误传播，支持 fatal/degraded 路由
- 兼容 OpenTelemetry 的追踪导出
- 统一 JSON 解析器，带 schema 校验

包含可观测性（执行追踪与导出）和质量评估（Agent 输出评分）模块。
"""

__version__ = "0.3.0"
