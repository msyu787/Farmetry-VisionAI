"""Lettuce growth recommendation + Ridge prediction + env optimization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[1] / "data" / "lettuce_processed"

# 분석에서 반복적으로 중요했던 환경 지표 (해석 가능)
KEY_FEATURES = [
    "AirHum_pct_min",
    "AirTemp_C_min",
    "WaterTemp_C_min",
    "Volume_L_mean",
    "AirTemp_C_delta_abs_mean",
    "AirHum_pct_daily_net_change",
    "Volume_L_daily_net_change",
    "EC_dS_m_mean",
]

RULE_THRESHOLDS = {
    "AirHum_pct_min": {"op": "lt", "value": 40.0, "action": "새벽 최저 습도를 40% 이상 유지하세요."},
    "WaterTemp_C_min": {"op": "lt", "value": 18.0, "action": "수온 최저치를 18°C 이상으로 올리세요."},
    "Volume_L_mean": {"op": "gt", "value": 60.0, "action": "양액 탱크가 가득 찬 상태입니다. 보충 타이밍을 조절하세요."},
    "AirTemp_C_delta_abs_mean": {
        "op": "gt",
        "value": 0.75,
        "action": "기온 시간당 요동이 큽니다. 온도를 안정적으로 유지하세요.",
    },
}


@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_cols: list[str]
    train_df: pd.DataFrame
    metrics: dict[str, float]
    cv_predictions: pd.DataFrame


def load_merged() -> pd.DataFrame:
    path = OUT / "daily_merged_bio_env.csv"
    if not path.exists():
        from lettuce_growth_env_analysis import main as rebuild

        rebuild()
    df = pd.read_csv(path).sort_values("DAT").reset_index(drop=True)
    return df


def build_lag_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """전날 환경 → 다음날 생장량 예측용."""
    lag = df.copy()
    for col in KEY_FEATURES:
        if col in lag.columns:
            lag[f"prev_{col}"] = lag[col].shift(1)
    lag["target"] = lag["biomass_delta_g_mean"]
    feature_cols = [f"prev_{c}" for c in KEY_FEATURES if f"prev_{c}" in lag.columns]
    return lag.dropna(subset=["target", *feature_cols]).reset_index(drop=True)


def evaluate_model(X: pd.DataFrame, y: pd.Series, alpha: float = 10.0) -> ModelBundle:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    loo = LeaveOneOut()
    y_pred = cross_val_predict(pipe, X, y, cv=loo)

    metrics = {
        "MAE_g": float(mean_absolute_error(y, y_pred)),
        "RMSE_g": float(np.sqrt(mean_squared_error(y, y_pred))),
        "R2": float(r2_score(y, y_pred)),
        "n_samples": int(len(y)),
    }
    metrics["MAPE_pct"] = float(
        np.mean(np.abs((y - y_pred) / y.replace(0, np.nan))) * 100
    )

    pipe.fit(X, y)
    cv_df = pd.DataFrame(
        {
            "actual_g": y.values,
            "predicted_g": y_pred,
            "error_g": y.values - y_pred,
        }
    )
    return ModelBundle(
        pipeline=pipe,
        feature_cols=list(X.columns),
        train_df=pd.DataFrame(),
        metrics=metrics,
        cv_predictions=cv_df,
    )


def train_growth_model(df: pd.DataFrame, alpha: float = 10.0) -> tuple[ModelBundle, pd.DataFrame]:
    lag = build_lag_dataset(df)
    feature_cols = [c for c in lag.columns if c.startswith("prev_")]
    X = lag[feature_cols]
    y = lag["target"]
    bundle = evaluate_model(X, y, alpha=alpha)
    bundle.train_df = lag
    bundle.cv_predictions = pd.DataFrame(
        {
            "md": lag["md"].values,
            "DAT": lag["DAT"].values,
            "actual_g": y.values,
            "predicted_g": bundle.cv_predictions["predicted_g"].values,
            "error_g": y.values - bundle.cv_predictions["predicted_g"].values,
        }
    )
    return bundle, lag


def recommend_rules(env_row: pd.Series | dict) -> list[dict[str, Any]]:
    if isinstance(env_row, pd.Series):
        row = env_row
    else:
        row = pd.Series(env_row)

    recs = []
    for feat, rule in RULE_THRESHOLDS.items():
        if feat not in row.index:
            continue
        val = row[feat]
        if pd.isna(val):
            continue
        triggered = (rule["op"] == "lt" and val < rule["value"]) or (
            rule["op"] == "gt" and val > rule["value"]
        )
        if triggered:
            recs.append(
                {
                    "feature": feat,
                    "current": round(float(val), 2),
                    "threshold": rule["value"],
                    "action": rule["action"],
                }
            )
    if not recs:
        recs.append({"feature": None, "current": None, "threshold": None, "action": "현재 환경은 규칙 기준 정상 범위입니다."})
    return recs


def optimize_environment(
    bundle: ModelBundle,
    lag_df: pd.DataFrame,
    n_grid: int = 5,
) -> pd.DataFrame:
    """관측 범위 내 grid search로 예측 생장량 최대 환경 탐색."""
    feature_cols = bundle.feature_cols
    # prev_* 컬럼의 관측 범위
    ranges = {}
    for col in feature_cols:
        lo, hi = lag_df[col].quantile(0.1), lag_df[col].quantile(0.9)
        ranges[col] = np.linspace(lo, hi, n_grid)

    # 핵심 4개만 조합, 나머지는 중앙값 고정 (조합 폭발 방지)
    tune_cols = [c for c in feature_cols if any(k in c for k in ["AirHum_pct", "WaterTemp", "Volume_L", "AirTemp_C"])]
    tune_cols = tune_cols[:4]
    fixed = {c: lag_df[c].median() for c in feature_cols if c not in tune_cols}

    rows = []
    for a in ranges[tune_cols[0]]:
        for b in ranges[tune_cols[1]]:
            for c in ranges[tune_cols[2]]:
                for d in ranges[tune_cols[3]]:
                    sample = {**fixed, tune_cols[0]: a, tune_cols[1]: b, tune_cols[2]: c, tune_cols[3]: d}
                    x = pd.DataFrame([sample])[feature_cols]
                    pred = float(bundle.pipeline.predict(x)[0])
                    rows.append({**sample, "predicted_growth_g": pred})

    result = pd.DataFrame(rows).sort_values("predicted_growth_g", ascending=False)
    return result.head(20).reset_index(drop=True)


def run_all(alpha: float = 10.0) -> dict[str, Any]:
    df = load_merged()
    bundle, lag = train_growth_model(df, alpha=alpha)

    # 마지막 날 환경 기준 규칙 추천
    last_env = df.iloc[-1]
    recommendations = recommend_rules(last_env)

    optimal = optimize_environment(bundle, lag)

    bundle.cv_predictions.to_csv(OUT / "model_cv_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([bundle.metrics]).to_csv(OUT / "model_metrics.csv", index=False, encoding="utf-8-sig")
    optimal.to_csv(OUT / "optimal_env_grid_top20.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(recommendations).to_csv(OUT / "rule_recommendations_latest.csv", index=False, encoding="utf-8-sig")

    return {
        "bundle": bundle,
        "lag_df": lag,
        "recommendations": recommendations,
        "optimal": optimal,
        "merged": df,
    }


if __name__ == "__main__":
    out = run_all()
    m = out["bundle"].metrics
    print("=== 모델 정확도 (Leave-One-Out CV) ===")
    for k, v in m.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n=== 규칙 기반 추천 (마지막 날) ===")
    for r in out["recommendations"]:
        print(f"  - {r['action']}")
    print("\n=== 예측 생장 최대 환경 (상위 3) ===")
    print(out["optimal"].head(3).to_string())
