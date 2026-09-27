#!/usr/bin/env python3
"""
Integration validation script - runs after all agents complete their tasks.

This script performs cross-module integration checks, runs test suites,
and generates a comprehensive validation report.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any


class Colors:
    """ANSI colors for section headers in the validation report."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    END = "\033[0m"


def print_section(title: str) -> None:
    """Print a colored section banner."""
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.END}\n")


def run_command(cmd: list[str], description: str) -> tuple[bool, str]:
    """Run a command and return (success, output)."""
    print(f"{Colors.YELLOW}▶ {description}...{Colors.END}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        if success:
            print(f"{Colors.GREEN}✓ {description} passed{Colors.END}")
        else:
            print(f"{Colors.RED}✗ {description} failed{Colors.END}")
            print(f"  Output: {output[:500]}")
        return success, output
    except subprocess.TimeoutExpired:
        print(f"{Colors.RED}✗ {description} timed out{Colors.END}")
        return False, "Timeout"
    except Exception as e:
        print(f"{Colors.RED}✗ {description} error: {e}{Colors.END}")
        return False, str(e)


def check_file_exists(path: Path) -> bool:
    """Check if a critical file exists."""
    exists = path.exists()
    status = f"{Colors.GREEN}✓{Colors.END}" if exists else f"{Colors.RED}✗{Colors.END}"
    print(f"  {status} {path}")
    return exists


def validate_modules() -> dict[str, Any]:
    """Validate all new modules can be imported."""
    print_section("Module Import Validation")

    modules = [
        "cuda_sft_qgen.observability",
        "cuda_sft_qgen.planner_v2",
        "cuda_sft_qgen.executor",
        "cuda_sft_qgen.graph.nodes.review_parallel",
        "cuda_sft_qgen.taxonomy.health",
        "cuda_sft_qgen.taxonomy.expander",
        "cuda_sft_qgen.config_v2",
    ]

    results = {}
    for module in modules:
        try:
            __import__(module)
            print(f"{Colors.GREEN}✓{Colors.END} {module}")
            results[module] = True
        except ImportError as e:
            print(f"{Colors.RED}✗{Colors.END} {module}: {e}")
            results[module] = False

    return results


def validate_files() -> dict[str, bool]:
    """Check all expected files were created."""
    print_section("File Structure Validation")

    root = Path(__file__).parent
    expected_files = [
        # New modules
        "src/cuda_sft_qgen/observability.py",
        "src/cuda_sft_qgen/planner_v2.py",
        "src/cuda_sft_qgen/executor.py",
        "src/cuda_sft_qgen/graph/nodes/review_parallel.py",
        "src/cuda_sft_qgen/taxonomy/health.py",
        "src/cuda_sft_qgen/taxonomy/expander.py",
        "src/cuda_sft_qgen/config_v2.py",
        # Tests
        "tests/test_observability.py",
        "tests/test_planner_adaptive.py",
        "tests/test_priority_executor.py",
        "tests/nodes/test_review_chair_parallel.py",
        "tests/taxonomy/test_health_analyzer.py",
        "tests/integration/test_full_pipeline.py",
        # Docs
        "docs/ARCHITECTURE.md",
        "docs/BEST_PRACTICES.md",
        "CHANGELOG.md",
        # Config
        "configs/planner.yaml",
        ".github/workflows/test.yml",
    ]

    results = {}
    for file_path in expected_files:
        full_path = root / file_path
        results[file_path] = check_file_exists(full_path)

    return results


def run_unit_tests() -> bool:
    """Run all unit tests."""
    print_section("Unit Tests")
    success, _ = run_command(
        ["pytest", "tests/", "-v", "--tb=short", "-x"],
        "Running unit tests"
    )
    return success


def run_integration_tests() -> bool:
    """Run integration tests with fake LLM."""
    print_section("Integration Tests")
    success, _ = run_command(
        ["pytest", "tests/integration/", "--fake-llm", "-v"],
        "Running integration tests"
    )
    return success


def check_coverage() -> tuple[bool, float]:
    """Run coverage analysis."""
    print_section("Code Coverage")
    success, output = run_command(
        [
            "pytest",
            "--cov=src/cuda_sft_qgen",
            "--cov-report=term",
            "--cov-report=html",
            "-q",
        ],
        "Measuring code coverage"
    )

    # Parse coverage percentage
    coverage_pct = 0.0
    for line in output.split("\n"):
        if "TOTAL" in line and "%" in line:
            parts = line.split()
            for part in parts:
                if "%" in part:
                    coverage_pct = float(part.rstrip("%"))
                    break

    print(f"\n  Coverage: {coverage_pct:.1f}%")
    target = 80.0
    if coverage_pct >= target:
        print(f"  {Colors.GREEN}✓ Coverage meets target ({target}%){Colors.END}")
    else:
        print(f"  {Colors.YELLOW}⚠ Coverage below target ({target}%){Colors.END}")

    return success, coverage_pct


def validate_cli_commands() -> dict[str, bool]:
    """Test new CLI commands."""
    print_section("CLI Command Validation")

    commands = [
        (["cuda-sft-qgen", "--help"], "Main CLI help"),
        (["cuda-sft-qgen", "validate-config"], "Config validation"),
        (["cuda-sft-qgen", "taxonomy", "health"], "Taxonomy health"),
    ]

    results = {}
    for cmd, desc in commands:
        success, _ = run_command(cmd, desc)
        results[desc] = success

    return results


def generate_report(results: dict[str, Any]) -> None:
    """Generate final validation report."""
    print_section("Validation Report")

    total_checks = 0
    passed_checks = 0

    # Module imports
    module_results = results.get("modules", {})
    total_checks += len(module_results)
    passed_checks += sum(module_results.values())

    # File existence
    file_results = results.get("files", {})
    total_checks += len(file_results)
    passed_checks += sum(file_results.values())

    # Tests
    if results.get("unit_tests"):
        passed_checks += 1
    total_checks += 1

    if results.get("integration_tests"):
        passed_checks += 1
    total_checks += 1

    # CLI
    cli_results = results.get("cli", {})
    total_checks += len(cli_results)
    passed_checks += sum(cli_results.values())

    coverage = results.get("coverage", 0.0)

    print(f"\n{'─' * 60}")
    print(f"Total Checks:    {total_checks}")
    print(f"Passed:          {Colors.GREEN}{passed_checks}{Colors.END}")
    print(f"Failed:          {Colors.RED}{total_checks - passed_checks}{Colors.END}")
    print(f"Success Rate:    {100 * passed_checks / max(total_checks, 1):.1f}%")
    print(f"Code Coverage:   {coverage:.1f}%")
    print(f"{'─' * 60}\n")

    if passed_checks == total_checks and coverage >= 80.0:
        print(f"{Colors.GREEN}✓ All validations passed!{Colors.END}")
        return True
    else:
        print(f"{Colors.YELLOW}⚠ Some validations failed or coverage below target{Colors.END}")
        return False


def main() -> int:
    """Main validation orchestrator."""
    print(f"\n{Colors.BLUE}CUDA-SFT-QGEN Integration Validation{Colors.END}")
    print(f"{Colors.BLUE}Started: 2026-09-16{Colors.END}\n")

    results = {}

    # 1. Validate modules
    results["modules"] = validate_modules()

    # 2. Validate files
    results["files"] = validate_files()

    # 3. Run unit tests
    results["unit_tests"] = run_unit_tests()

    # 4. Run integration tests
    results["integration_tests"] = run_integration_tests()

    # 5. Check coverage
    _, coverage = check_coverage()
    results["coverage"] = coverage

    # 6. Validate CLI
    results["cli"] = validate_cli_commands()

    # 7. Generate report
    success = generate_report(results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
