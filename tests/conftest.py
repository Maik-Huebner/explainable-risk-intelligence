from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explainable_risk_intelligence.data.schema import RAW_COLUMNS, TARGET_SOURCE


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    generator = np.random.default_rng(42)
    rows = 400
    frame = pd.DataFrame(index=range(rows), columns=RAW_COLUMNS)
    frame["ID"] = np.arange(1, rows + 1)
    frame["LIMIT_BAL"] = generator.integers(10_000, 500_001, rows)
    frame["SEX"] = generator.integers(1, 3, rows)
    frame["EDUCATION"] = generator.integers(1, 5, rows)
    frame["MARRIAGE"] = generator.integers(1, 4, rows)
    frame["AGE"] = generator.integers(21, 75, rows)
    for column in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        frame[column] = generator.integers(-2, 5, rows)
    for index in range(1, 7):
        frame[f"BILL_AMT{index}"] = generator.integers(-5_000, 300_000, rows)
        frame[f"PAY_AMT{index}"] = generator.integers(0, 80_000, rows)
    logit = 0.7 * (frame["PAY_0"] > 0) + 0.3 * (frame["PAY_2"] > 0)
    probability = 1 / (1 + np.exp(-logit + 1.3))
    frame[TARGET_SOURCE] = generator.binomial(1, probability)
    return frame.astype({column: int for column in RAW_COLUMNS})
