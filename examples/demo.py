"""InsightFlow 演示 - 运行多智能体数据分析流水线。

本脚本演示完整的 InsightFlow 流水线：
1. 加载示例电商销售数据
2. 侦察智能体（Scout Agent）探索并分析数据概况
3. 清洗智能体（Cleaner Agent）创建并执行清洗计划（需人工审核）
4. 分析智能体（Analyst Agent）执行统计分析
5. 可视化智能体（Visualizer Agent）生成图表
6. 报告智能体（Reporter Agent）汇总所有内容生成 Markdown 报告

额外支持：
- 执行追踪：计时、输入/输出快照、可视化时间线（HTML）
- 质量评估：逐智能体评分、改进建议

用法：
    # 基本用法（使用示例数据，默认启用追踪和评估）
    python examples/demo.py

    # 自定义数据文件和分析任务
    python examples/demo.py --data path/to/data.csv --task "your analysis question"

    # 跳过人工审核（自动批准清洗计划）
    python examples/demo.py --auto

    # 禁用追踪或评估
    python examples/demo.py --no-trace
    python examples/demo.py --no-eval

前置条件：
    - 在 .env 文件或环境变量中设置 OPENAI_API_KEY
    - 安装依赖：pip install -e ".[dev]"
"""

import argparse
import os
import sys
import json
from pathlib import Path

# 将项目 src 目录添加到路径，便于开发调试
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insightflow.graph import run_pipeline, get_mermaid_diagram
from insightflow.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="InsightFlow: Multi-Agent Data Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/demo.py
  python examples/demo.py --task "分析各城市的销售总额分布"
  python examples/demo.py --data my_data.csv --task "分析销售趋势" --auto
        """,
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "examples" / "sample_data" / "sales_data.csv"),
        help="CSV 数据文件路径（默认：示例销售数据）",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="请分析这份电商销售数据，重点关注：1) 各产品类别的销售表现对比；2) 各城市的销售分布特征；3) 价格与评分之间的关系；4) 支付方式偏好分析",
        help="分析任务的自然语言描述",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自动批准清洗计划（跳过人工审核）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="报告和图表的输出目录（默认：output/）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="启用详细输出（默认：True）",
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="打印 Mermaid 工作流图并退出",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="禁用执行追踪（默认启用）",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="禁用质量评估（默认启用）",
    )

    args = parser.parse_args()

    # 显示工作流图模式
    if args.show_graph:
        print("\n📊 InsightFlow Workflow Diagram:\n")
        print(get_mermaid_diagram())
        return

    # 验证数据文件
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data file not found: {data_path}")
        sys.exit(1)

    # 加载配置（检查 OPENAI_API_KEY）
    config = load_config()
    if not config.llm.api_key:
        print("❌ Error: OPENAI_API_KEY not set.")
        print("   Please set it in a .env file or environment variable.")
        print("   See .env.example for reference.")
        sys.exit(1)

    # 覆盖输出目录配置
    config.pipeline.output_dir = args.output
    config.pipeline.human_review = not args.auto
    os.environ["OUTPUT_DIR"] = args.output

    # 运行流水线
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          🔬 InsightFlow — Multi-Agent Pipeline          ║")
    print("║    LangGraph + MCP + Multi-Agent Collaboration          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    try:
        result = run_pipeline(
            data_path=str(data_path),
            analysis_task=args.task,
            verbose=args.verbose,
            enable_trace=not args.no_trace,
            enable_eval=not args.no_eval,
            output_dir=args.output,
        )

        # 打印摘要
        print("\n📋 Pipeline Execution Summary:")
        print(f"   Agent messages: {len(result.get('messages', []))}")
        print(f"   Charts generated: {len(result.get('charts', []))}")
        print(f"   Errors: {len(result.get('errors', []))}")
        print(f"   Iterations: {result.get('iteration', 0)}")

        if result.get("charts"):
            print("\n📈 Generated Charts:")
            for chart in result["charts"]:
                print(f"   - {chart}")

        if result.get("report"):
            report_path = Path(config.pipeline.output_dir) / "analysis_report.md"
            print(f"\n📝 Report saved to: {report_path}")

        if result.get("errors"):
            print("\n⚠️  Errors encountered:")
            for err in result["errors"]:
                print(f"   - {err[:100]}...")

        # 保存状态快照，用于调试
        state_path = Path(config.pipeline.output_dir) / "pipeline_state.json"
        state_snapshot = {
            "data_path": result.get("data_path", ""),
            "analysis_task": result.get("analysis_task", ""),
            "data_profile": result.get("data_profile", {}),
            "cleaning_plan": result.get("cleaning_plan", {}),
            "analysis_results": result.get("analysis_results", {}),
            "charts": result.get("charts", []),
            "iteration": result.get("iteration", 0),
            "messages_count": len(result.get("messages", [])),
            "errors": result.get("errors", []),
        }
        state_path.write_text(
            json.dumps(state_snapshot, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n💾 State snapshot saved to: {state_path}")

        # 提示追踪和评估输出文件
        out = Path(args.output)
        if not args.no_trace:
            print(f"🔍 Trace exported to: {out}/trace_*.json, trace_*.md, trace_*.html")
        if not args.no_eval:
            print(f"📊 Evaluation exported to: {out}/evaluation.json, {out}/evaluation.md")

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
