#!/usr/bin/env python3
"""
Benchmark Comparison Script

Generates a comparison report from benchmark metrics JSON files.
Reports effect sizes and statistical significance.
"""

import json
import sys
from pathlib import Path
from typing import Any

def load_metrics(path: Path) -> dict[str, Any]:
    """Load metrics from JSON file."""
    with open(path) as f:
        return json.load(f)

def format_ci(mean: float, ci_lower: float, ci_upper: float) -> str:
    """Format mean with confidence interval."""
    return f"{mean:.4g} [{ci_lower:.4g}, {ci_upper:.4g}]"

def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def generate_report(metrics_dir: Path, output_path: Path) -> None:
    """Generate comparison report."""
    report = ["# Benchmark Comparison Report\n"]
    report.append(f"Generated from metrics in `{metrics_dir}`\n")

    # Load all metrics files
    metrics_files = list(metrics_dir.glob("*_benchmark.json"))

    for metrics_file in sorted(metrics_files):
        metrics = load_metrics(metrics_file)

        report.append(f"\n## {metrics['benchmark']}\n")
        report.append(f"Version: {metrics['version']}\n")
        report.append(f"Hardware: {metrics['hardware']['cpu']}\n\n")

        report.append("| Metric | Mean [95% CI] | Effect Size | Status |\n")
        report.append("|--------|---------------|-------------|--------|\n")

        for name, result in metrics.get("results", {}).items():
            mean = result.get("mean", 0)
            ci_lower = result.get("ci_lower", mean)
            ci_upper = result.get("ci_upper", mean)

            ci_str = format_ci(mean, ci_lower, ci_upper)

            effect = result.get("effect_size", {})
            d = effect.get("cohens_d", "-")
            if isinstance(d, (int, float)):
                effect_str = f"d={d:.2f} ({interpret_effect_size(d)})"
            else:
                effect_str = "-"

            passed = result.get("passed", None)
            if passed is True:
                status = "✅ Pass"
            elif passed is False:
                status = "❌ Fail"
            else:
                status = "-"

            report.append(f"| {name} | {ci_str} | {effect_str} | {status} |\n")

    report.append("\n## Summary\n")
    report.append("All benchmarks include 95% confidence intervals.\n")
    report.append("Effect sizes interpreted using Cohen's conventions.\n")

    output_path.write_text("".join(report))
    print(f"Report written to {output_path}")

def main() -> int:
    metrics_dir = Path("metrics")
    output_path = Path("reports/benchmark_comparison.md")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        generate_report(metrics_dir, output_path)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
