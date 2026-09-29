"""Leakage-safe scikit-learn pipelines."""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from explainable_risk_intelligence.features.build import MODEL_FEATURES


def make_preprocessor() -> ColumnTransformer:
    """Fit imputation and scaling only within the estimator pipeline."""

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric, MODEL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline(steps=[("preprocessor", make_preprocessor()), ("model", estimator)])


def candidate_models(seed: int, fast: bool = False) -> dict[str, Pipeline]:
    trees = 80 if fast else 400
    iterations = 80 if fast else 250
    return {
        "dummy_prior": make_model_pipeline(DummyClassifier(strategy="prior")),
        "logistic_regression": make_model_pipeline(
            LogisticRegression(max_iter=1_500, class_weight="balanced", random_state=seed)
        ),
        "random_forest": make_model_pipeline(
            RandomForestClassifier(
                n_estimators=trees,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )
        ),
        "hist_gradient_boosting": make_model_pipeline(
            HistGradientBoostingClassifier(
                max_iter=iterations,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=seed,
            )
        ),
    }


def hgb_search_space(fast: bool = False) -> dict[str, list[object]]:
    if fast:
        return {
            "model__learning_rate": [0.08],
            "model__max_leaf_nodes": [15],
            "model__min_samples_leaf": [20],
            "model__l2_regularization": [0.1],
            "model__max_iter": [60],
        }
    return {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 40, 80],
        "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
        "model__max_iter": [150, 250, 350],
    }
