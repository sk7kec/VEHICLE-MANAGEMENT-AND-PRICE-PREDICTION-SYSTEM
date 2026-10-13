"""
Model Evaluation Report Generator

Loads trained model artifacts and generates a comprehensive evaluation report.

Author: Sahishnakumaar S S (24ADR134)
"""

import os
import json
import sys


def generate_report(artifacts_dir, output_path=None):
    """Generate a markdown evaluation report from model metadata."""

    price_meta_path = os.path.join(artifacts_dir, "price_model_metadata.json")
    category_meta_path = os.path.join(artifacts_dir, "category_model_metadata.json")

    report_lines = []
    report_lines.append("# Model Evaluation Report")
    report_lines.append("")
    report_lines.append("**Project:** Vehicle Management and Price Prediction System")
    report_lines.append("**Author:** Sahishnakumaar S S (24ADR134)")
    report_lines.append("")

    # Price model
    if os.path.exists(price_meta_path):
        with open(price_meta_path, "r", encoding="utf-8") as f:
            pm = json.load(f)

        report_lines.append("## 1. Price Prediction Model (Regression)")
        report_lines.append("")
        report_lines.append(f"- **Model Type:** {pm.get('model_type', 'N/A')}")
        report_lines.append(f"- **Training Date:** {pm.get('training_date', 'N/A')}")
        sources = pm.get('dataset_sources', [pm.get('dataset_path', 'N/A')])
        report_lines.append(f"- **Dataset:** {', '.join(sources) if isinstance(sources, list) else sources} ({pm.get('dataset_type', 'N/A')})")
        report_lines.append(f"- **Total Records:** {pm.get('total_records', 'N/A')}")
        report_lines.append(f"- **Train/Test Split:** {pm.get('train_size', 'N/A')} / {pm.get('test_size', 'N/A')}")
        report_lines.append(f"- **Random Seed:** {pm.get('random_seed', 'N/A')}")
        report_lines.append("")

        metrics = pm.get("evaluation_metrics", {})
        report_lines.append("### Evaluation Metrics")
        report_lines.append("")
        report_lines.append(f"| Metric | Value |")
        report_lines.append(f"|---|---|")
        report_lines.append(f"| MAE | ₹{metrics.get('mae', 'N/A'):,.2f} |")
        report_lines.append(f"| RMSE | ₹{metrics.get('rmse', 'N/A'):,.2f} |")
        report_lines.append(f"| R² Score | {metrics.get('r2_score', 'N/A')} |")
        report_lines.append("")

        hp = pm.get("hyperparameters", {})
        report_lines.append("### Hyperparameters")
        report_lines.append("")
        for k, v in hp.items():
            report_lines.append(f"- **{k}:** {v}")
        report_lines.append("")

        report_lines.append(f"> **Disclaimer:** {pm.get('disclaimer', '')}")
        report_lines.append("")
    else:
        report_lines.append("## 1. Price Prediction Model")
        report_lines.append("*Model metadata not found. Run training first.*")
        report_lines.append("")

    # Category model
    if os.path.exists(category_meta_path):
        with open(category_meta_path, "r", encoding="utf-8") as f:
            cm = json.load(f)

        report_lines.append("## 2. Category Classification Model")
        report_lines.append("")
        report_lines.append(f"- **Model Type:** {cm.get('model_type', 'N/A')}")
        report_lines.append(f"- **Training Date:** {cm.get('training_date', 'N/A')}")
        sources = cm.get('dataset_sources', [cm.get('dataset_path', 'N/A')])
        report_lines.append(f"- **Dataset:** {', '.join(sources) if isinstance(sources, list) else sources} ({cm.get('dataset_type', 'N/A')})")
        report_lines.append(f"- **Categories:** {', '.join(cm.get('categories', []))}")
        report_lines.append(f"- **Total Records:** {cm.get('total_records', 'N/A')}")
        report_lines.append(f"- **Train/Test Split:** {cm.get('train_size', 'N/A')} / {cm.get('test_size', 'N/A')}")
        report_lines.append("")

        boundaries = cm.get("category_boundaries", {})
        report_lines.append("### Category Definitions")
        report_lines.append("")
        for cat, boundary in boundaries.items():
            report_lines.append(f"- **{cat}:** {boundary}")
        report_lines.append("")
        report_lines.append(f"*{cm.get('labeling_methodology', '')}*")
        report_lines.append("")

        metrics = cm.get("evaluation_metrics", {})
        report_lines.append("### Overall Metrics")
        report_lines.append("")
        report_lines.append(f"| Metric | Value |")
        report_lines.append(f"|---|---|")
        report_lines.append(f"| Accuracy | {metrics.get('accuracy', 'N/A')} |")
        report_lines.append(f"| Precision (weighted) | {metrics.get('precision_weighted', 'N/A')} |")
        report_lines.append(f"| Recall (weighted) | {metrics.get('recall_weighted', 'N/A')} |")
        report_lines.append(f"| F1-Score (weighted) | {metrics.get('f1_score_weighted', 'N/A')} |")
        report_lines.append("")

        per_class = metrics.get("per_class", {})
        if per_class:
            report_lines.append("### Per-Class Metrics")
            report_lines.append("")
            report_lines.append("| Category | Precision | Recall | F1-Score | Support |")
            report_lines.append("|---|---|---|---|---|")
            for cat, m in per_class.items():
                report_lines.append(
                    f"| {cat} | {m.get('precision', 'N/A')} | {m.get('recall', 'N/A')} | {m.get('f1_score', 'N/A')} | {m.get('support', 'N/A')} |"
                )
            report_lines.append("")

        report_lines.append(f"> **Disclaimer:** {cm.get('disclaimer', '')}")
        report_lines.append("")
    else:
        report_lines.append("## 2. Category Classification Model")
        report_lines.append("*Model metadata not found. Run training first.*")
        report_lines.append("")

    report = "\n".join(report_lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Evaluation report saved to: {output_path}")
    else:
        print(report)

    return report


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    artifacts_dir = os.path.join(project_root, "artifacts")
    output_path = os.path.join(artifacts_dir, "evaluation_report.md")

    generate_report(artifacts_dir, output_path)
