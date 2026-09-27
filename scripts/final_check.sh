#!/bin/bash

echo "========================================"
echo "CUDA-SFT-QGEN 最终状态检查"
echo "========================================"
echo ""

echo "📦 核心模块检查："
echo ""

modules=(
    "src/cuda_sft_qgen/observability.py"
    "src/cuda_sft_qgen/planner_v2.py"
    "src/cuda_sft_qgen/executor.py"
    "src/cuda_sft_qgen/config_v2.py"
    "src/cuda_sft_qgen/taxonomy_health/health.py"
    "src/cuda_sft_qgen/taxonomy_health/expander.py"
)

for module in "${modules[@]}"; do
    if [ -f "$module" ]; then
        lines=$(wc -l < "$module")
        echo "✅ $module ($lines 行)"
    else
        echo "❌ $module (缺失)"
    fi
done

echo ""
echo "📚 文档检查："
echo ""

docs=(
    "docs/ARCHITECTURE.md"
    "docs/BEST_PRACTICES.md"
    "docs/CONFIG_MIGRATION.md"
    "CHANGELOG.md"
    "AUDIT_PLAN.md"
    "SUMMARY.md"
)

for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        lines=$(wc -l < "$doc")
        echo "✅ $doc ($lines 行)"
    else
        echo "❌ $doc (缺失)"
    fi
done

echo ""
echo "⚙️ 配置文件检查："
echo ""

configs=(
    "configs/planner.yaml"
    "configs/profiles/dev.yaml"
    "configs/profiles/test.yaml"
    "configs/profiles/prod.yaml"
)

for config in "${configs[@]}"; do
    if [ -f "$config" ]; then
        echo "✅ $config"
    else
        echo "❌ $config (缺失)"
    fi
done

echo ""
echo "🧪 测试文件统计："
echo ""

unit_tests=$(find tests/ -name "test_*.py" -not -path "*/integration/*" -not -path "*/e2e/*" 2>/dev/null | wc -l)
integration_tests=$(find tests/integration -name "test_*.py" 2>/dev/null | wc -l)
e2e_tests=$(find tests/e2e -name "test_*.py" 2>/dev/null | wc -l)

echo "单元测试: $unit_tests 个"
echo "集成测试: $integration_tests 个"
echo "端到端测试: $e2e_tests 个"
echo "总计: $((unit_tests + integration_tests + e2e_tests)) 个"

echo ""
echo "🛠️ 工具脚本检查："
echo ""

tools=(
    "scripts/validate_integration.py"
    "scripts/monitor_agents.py"
    "scripts/final_check.sh"
    "Makefile"
)

for tool in "${tools[@]}"; do
    if [ -f "$tool" ]; then
        echo "✅ $tool"
    else
        echo "❌ $tool (缺失)"
    fi
done

echo ""
echo "📊 代码统计："
echo ""

total_py_lines=$(find src/cuda_sft_qgen -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
test_py_lines=$(find tests -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
doc_lines=$(find docs -name "*.md" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
root_doc_lines=$(wc -l *.md 2>/dev/null | tail -1 | awk '{print $1}')

echo "源代码: $total_py_lines 行"
echo "测试代码: $test_py_lines 行"
echo "文档 (docs/): $doc_lines 行"
echo "文档 (根目录): $root_doc_lines 行"

echo ""
echo "========================================"
echo "检查完成！"
echo "========================================"
