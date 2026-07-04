"""InsightFlow的所有代理模块。

导出所有 LangGraph 节点函数，方便在图定义中直接使用：

- ``scout_node``           -- 数据加载与画像（数据侦察）
- ``cleaner_plan_node``    -- 清洗计划生成（清洗计划）
- ``cleaner_execute_node`` -- 清洗计划执行（清洗执行）
- ``analyst_node``         -- 统计分析（数据分析）
- ``visualizer_node``      -- 图表生成（可视化生成）
- ``reporter_node``        -- 报告生成（报告撰写）
"""

from insightflow.agents.scout import scout_node
from insightflow.agents.cleaner import cleaner_plan_node, cleaner_execute_node
from insightflow.agents.analyst import analyst_node
from insightflow.agents.visualizer import visualizer_node
from insightflow.agents.reporter import reporter_node

__all__ = [
    "scout_node",
    "cleaner_plan_node",
    "cleaner_execute_node",
    "analyst_node",
    "visualizer_node",
    "reporter_node",
]
