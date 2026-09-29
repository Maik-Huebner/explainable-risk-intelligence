"""Command-line entry points for the complete project lifecycle."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from explainable_risk_intelligence.data.download import download_dataset
from explainable_risk_intelligence.data.prepare import prepare_data
from explainable_risk_intelligence.evaluation.metrics import calculate_metrics
from explainable_risk_intelligence.evaluation.report import generate_reports
from explainable_risk_intelligence.evaluation.run import evaluate_model
from explainable_risk_intelligence.fairness.metrics import audit_fairness
from explainable_risk_intelligence.models.pipeline import make_model_pipeline
from explainable_risk_intelligence.models.train import train_models
from explainable_risk_intelligence.monitoring.drift import population_stability_index


def smoke_test() -> None:
    """Run an offline, synthetic package-level model smoke test for CI."""

    generator = np.random.default_rng(42)
    rows = 240
    columns = [
        "LIMIT_BAL",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        *[f"BILL_AMT{i}" for i in range(1, 7)],
        *[f"PAY_AMT{i}" for i in range(1, 7)],
        "avg_bill_amount",
        "avg_payment_amount",
        "credit_utilization",
        "payment_to_bill_ratio",
        "max_payment_delay",
        "delayed_month_count",
        "bill_amount_trend",
        "payment_amount_trend",
    ]
    X = pd.DataFrame(generator.normal(size=(rows, len(columns))), columns=columns)
    X["LIMIT_BAL"] = generator.uniform(10_000, 500_000, rows)
    y = (X["PAY_0"] + generator.normal(size=rows) > 0).astype(int).to_numpy()
    model = make_model_pipeline(LogisticRegression(max_iter=500, random_state=42))
    model.fit(X.iloc[:180], y[:180])
    probabilities = model.predict_proba(X.iloc[180:])[:, 1]
    metrics = calculate_metrics(y[180:], probabilities)
    audit_frame = pd.DataFrame(
        {
            "default_next_month": y[180:],
            "audit_sex": np.where(np.arange(60) % 2, "female", "male"),
        }
    )
    audit_fairness(audit_frame, probabilities, 0.5, ["audit_sex"])
    psi = population_stability_index(X["LIMIT_BAL"].iloc[:180], X["LIMIT_BAL"].iloc[180:])
    if not 0 <= metrics["roc_auc"] <= 1 or psi < 0:
        raise RuntimeError("Smoke test produced invalid metrics")
    print("Synthetic model smoke test passed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="risk-intelligence",
        description="Reproduce and inspect the Explainable Risk Intelligence evidence package.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("download-data", help="Download and verify the official UCI archive")
    subcommands.add_parser("prepare-data", help="Validate, engineer, and split the dataset")
    train = subcommands.add_parser(
        "train", help="Train and select models without loading test data"
    )
    train.add_argument("--fast", action="store_true")
    evaluate = subcommands.add_parser(
        "evaluate", help="Evaluate the locked model on final test data"
    )
    evaluate.add_argument("--fast", action="store_true")
    subcommands.add_parser("report", help="Refresh human-readable reports from evidence artifacts")
    reproduce = subcommands.add_parser("reproduce", help="Run the complete evidence pipeline")
    reproduce.add_argument("--fast", action="store_true")
    subcommands.add_parser("smoke-test", help="Run an offline synthetic package smoke test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download-data":
        path = download_dataset()
        print(f"Verified dataset: {path}")
    elif args.command == "prepare-data":
        splits = prepare_data()
        print("Prepared splits:", {name: len(frame) for name, frame in splits.items()})
    elif args.command == "train":
        metadata = train_models(fast=args.fast)
        print("Selected model:", metadata["selected_model"])
    elif args.command == "evaluate":
        result = evaluate_model(fast=args.fast)
        print("Final test metrics:", result["metrics"])
    elif args.command == "report":
        generate_reports()
        print("Reports generated.")
    elif args.command == "reproduce":
        download_dataset()
        prepare_data()
        train_models(fast=args.fast)
        evaluate_model(fast=args.fast)
        generate_reports()
        print("Full reproduction completed.")
    elif args.command == "smoke-test":
        smoke_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
