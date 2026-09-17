#!/usr/bin/env python3
"""Read IMPLEMENTATION_STATUS.md and print a compact agent-progress table."""

import json
import time
from datetime import datetime
from pathlib import Path


def get_agent_status():
    """Read agent status from IMPLEMENTATION_STATUS.md and task outputs."""

    agents = {
        "agent-architecture": {
            "name": "架构重构与状态管理",
            "status": "🔄 运行中",
            "files_expected": [
                "src/cuda_sft_qgen/graph/state_v2.py",
                "src/cuda_sft_qgen/observability.py",
                "tests/graph/test_state_validation.py",
                "tests/test_observability.py",
            ]
        },
        "agent-planner": {
            "name": "Planner 智能调度升级",
            "status": "🔄 运行中",
            "files_expected": [
                "src/cuda_sft_qgen/planner_v2.py",
                "configs/planner.yaml",
                "tests/test_planner_adaptive.py",
            ]
        },
        "agent-writer": {
            "name": "Writer 并发优化",
            "status": "🔄 运行中",
            "files_expected": [
                "src/cuda_sft_qgen/executor.py",
                "tests/test_priority_executor.py",
            ]
        },
        "agent-review": {
            "name": "Review Chair 并行化",
            "status": "🔄 运行中",
            "files_expected": [
                "src/cuda_sft_qgen/graph/nodes/review_parallel.py",
                "tests/nodes/test_review_chair_parallel.py",
            ]
        },
        "agent-taxonomy": {
            "name": "Taxonomy 健康分析",
            "status": "🔄 运行中",
            "files_expected": [
                "src/cuda_sft_qgen/taxonomy/health.py",
                "src/cuda_sft_qgen/taxonomy/expander.py",
                "tests/taxonomy/test_health_analyzer.py",
            ]
        },
        "agent-testing": {
            "name": "测试体系完善",
            "status": "🔄 运行中",
            "files_expected": [
                "tests/integration/test_full_pipeline.py",
                "tests/e2e/test_cli_generate.py",
                ".github/workflows/test.yml",
            ]
        },
        "agent-docs": {
            "name": "配置管理与文档更新",
            "status": "🔄 运行中",
            "files_expected": [
                "src/cuda_sft_qgen/config_v2.py",
                "docs/ARCHITECTURE.md",
                "docs/BEST_PRACTICES.md",
                "CHANGELOG.md",
            ]
        },
    }

    root = Path("/home/aa24aa/work/cuda-sft-qgen")

    # Check file completion
    for agent_id, info in agents.items():
        files_created = sum(
            1 for f in info["files_expected"]
            if (root / f).exists()
        )
        total_files = len(info["files_expected"])
        info["progress"] = f"{files_created}/{total_files}"

        if files_created == total_files:
            info["status"] = "✅ 完成"
        elif files_created > 0:
            info["status"] = f"🔄 进行中 ({files_created}/{total_files})"

    return agents


def print_status():
    """Print current status of all agents."""
    agents = get_agent_status()

    print("\n" + "=" * 80)
    print(f"CUDA-SFT-QGEN Agent 进度监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    completed = sum(1 for a in agents.values() if "✅" in a["status"])
    total = len(agents)

    print(f"总体进度: {completed}/{total} 完成\n")

    for agent_id, info in agents.items():
        print(f"{info['status']} {info['name']}")
        print(f"   进度: {info['progress']} 文件")
        print(f"   预期文件: {', '.join([Path(f).name for f in info['files_expected'][:2]])}")
        if len(info['files_expected']) > 2:
            print(f"             ...等 {len(info['files_expected']) - 2} 个文件")
        print()

    print("=" * 80)

    if completed == total:
        print("\n🎉 所有 Agent 已完成！可以开始集成验证。\n")
        print("下一步:")
        print("  1. 运行 make validate")
        print("  2. 检查 POST_AGENT_CHECKLIST.md")
        print("  3. 执行集成测试")
        return True
    else:
        print(f"\n⏳ 等待 {total - completed} 个 Agent 完成...\n")
        return False


def monitor_loop(interval=30):
    """Continuously monitor agent progress."""
    try:
        while True:
            all_done = print_status()
            if all_done:
                break
            print(f"下次刷新: {interval} 秒后... (Ctrl+C 退出)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n监控已停止。")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        print_status()
    else:
        monitor_loop()
