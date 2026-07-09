"""Model comparison: Ridge CV tuning + time-series baselines."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lettuce_growth_model import OUT, ModelBundle, build_lag_dataset, load_merged

# statsmodels optional for SARIMAX
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_s = pd.Series(y_true)
    return {
        "MAE_g": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_g": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_pct": float(np.mean(np.abs((y_s - y_pred) / y_s.replace(0, np.nan))) * 100),
        "within_1g_pct": float(np.mean(np.abs(y_true - y_pred) <= 1) * 100),
        "within_2g_pct": float(np.mean(np.abs(y_true - y_pred) <= 2) * 100),
        "n_samples": int(len(y_true)),
    }


def _loo_predict(estimator, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    return cross_val_predict(estimator, X, y, cv=LeaveOneOut())


def _prepare_xy(lag: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_cols = [c for c in lag.columns if c.startswith("prev_")]
    X = lag[feature_cols]
    y = lag["target"]
    return X, y, feature_cols


def eval_ridge_baseline(X: pd.DataFrame, y: pd.Series, alpha: float = 10.0) -> dict[str, Any]:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))])
    pred = _loo_predict(pipe, X, y)
    return {"model": f"Ridge (alpha={alpha})", "best_params": {"alpha": alpha}, "pred": pred, **_metrics(y, pred)}


def eval_ridge_tuned(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    grid = {"model__alpha": np.logspace(-2, 3, 30)}
    search = GridSearchCV(pipe, grid, cv=LeaveOneOut(), scoring="neg_mean_absolute_error", n_jobs=-1)
    search.fit(X, y)
    pred = cross_val_predict(search.best_estimator_, X, y, cv=LeaveOneOut())
    return {
        "model": "Ridge (CV tuned)",
        "best_params": search.best_params_,
        "pred": pred,
        **_metrics(y, pred),
    }


def eval_elasticnet_tuned(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(max_iter=5000))])
    grid = {
        "model__alpha": np.logspace(-2, 2, 15),
        "model__l1_ratio": [0.2, 0.5, 0.8, 1.0],
    }
    search = GridSearchCV(pipe, grid, cv=LeaveOneOut(), scoring="neg_mean_absolute_error", n_jobs=-1)
    search.fit(X, y)
    pred = cross_val_predict(search.best_estimator_, X, y, cv=LeaveOneOut())
    return {
        "model": "ElasticNet (CV tuned)",
        "best_params": search.best_params_,
        "pred": pred,
        **_metrics(y, pred),
    }


def eval_gbr_tuned(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", GradientBoostingRegressor(random_state=42))])
    grid = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [2, 3],
        "model__learning_rate": [0.05, 0.1],
    }
    search = GridSearchCV(pipe, grid, cv=LeaveOneOut(), scoring="neg_mean_absolute_error", n_jobs=-1)
    search.fit(X, y)
    pred = cross_val_predict(search.best_estimator_, X, y, cv=LeaveOneOut())
    return {
        "model": "GradientBoosting (CV tuned)",
        "best_params": search.best_params_,
        "pred": pred,
        **_metrics(y, pred),
    }


def eval_rf_tuned(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(random_state=42))])
    grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [2, 4],
    }
    search = GridSearchCV(pipe, grid, cv=LeaveOneOut(), scoring="neg_mean_absolute_error", n_jobs=-1)
    search.fit(X, y)
    pred = cross_val_predict(search.best_estimator_, X, y, cv=LeaveOneOut())
    return {
        "model": "RandomForest (CV tuned)",
        "best_params": search.best_params_,
        "pred": pred,
        **_metrics(y, pred),
    }


def eval_mean_baseline(y: pd.Series) -> dict[str, Any]:
    pred = np.full(len(y), y.mean())
    return {"model": "Baseline (평균 예측)", "best_params": {}, "pred": pred, **_metrics(y, pred)}


def eval_lag_growth_plus_env(lag: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """전날 생장량 + 전날 환경 (시계열 특성 추가)."""
    ext = X.copy()
    ext["prev_growth"] = lag["target"].shift(1)
    ext = ext.iloc[1:].reset_index(drop=True)
    y2 = y.iloc[1:].reset_index(drop=True)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    grid = {"model__alpha": np.logspace(-2, 3, 20)}
    search = GridSearchCV(pipe, grid, cv=LeaveOneOut(), scoring="neg_mean_absolute_error", n_jobs=-1)
    search.fit(ext, y2)
    pred = cross_val_predict(search.best_estimator_, ext, y2, cv=LeaveOneOut())
    return {
        "model": "Ridge + lag1 생장량 (CV tuned)",
        "best_params": search.best_params_,
        "pred": pred,
        **_metrics(y2, pred),
    }


def _loo_sarimax(y: pd.Series, exog: pd.DataFrame, order: tuple[int, int, int] = (1, 0, 0)) -> np.ndarray:
    preds = np.zeros(len(y))
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        y_train = y.values[mask]
        ex_train = exog.values[mask]
        ex_test = exog.values[i : i + 1]
        try:
            fit = SARIMAX(y_train, exog=ex_train, order=order, enforce_stationarity=False, enforce_invertibility=False)
            res = fit.fit(disp=False, maxiter=200)
            preds[i] = res.forecast(1, exog=ex_test)[0]
        except Exception:
            preds[i] = y_train.mean()
    return preds


def eval_sarimax(lag: pd.DataFrame, X: pd.DataFrame, y: pd.Series, exog_cols: list[str] | None = None) -> dict[str, Any]:
    if not HAS_STATSMODELS:
        return {"model": "SARIMAX", "skip": True}
    cols = exog_cols or [
        "prev_AirHum_pct_min",
        "prev_WaterTemp_C_min",
        "prev_Volume_L_mean",
    ]
    exog = X[cols]
    pred = _loo_sarimax(y, exog, order=(1, 0, 0))
    return {
        "model": "SARIMAX(1,0,0)+환경3종 (LOO)",
        "best_params": {"exog": cols, "order": (1, 0, 0)},
        "pred": pred,
        **_metrics(y, pred),
    }


def _loo_arima(y: pd.Series, order: tuple[int, int, int] = (1, 0, 1)) -> np.ndarray:
    preds = np.zeros(len(y))
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        y_train = y.values[mask]
        try:
            fit = SARIMAX(y_train, order=order, enforce_stationarity=False, enforce_invertibility=False)
            res = fit.fit(disp=False, maxiter=200)
            preds[i] = res.forecast(1)[0]
        except Exception:
            preds[i] = y_train.mean()
    return preds


def eval_arima_univariate(y: pd.Series) -> dict[str, Any]:
    if not HAS_STATSMODELS:
        return {"model": "ARIMA", "skip": True}
    pred = _loo_arima(y, order=(1, 0, 1))
    return {
        "model": "ARIMA(1,0,1) 생장만 (LOO)",
        "best_params": {"order": (1, 0, 1)},
        "pred": pred,
        **_metrics(y, pred),
    }


def _run_all_evaluations(lag: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> list[dict[str, Any]]:
    runners: list[Callable[[], dict[str, Any]]] = [
        lambda: eval_mean_baseline(y),
        lambda: eval_ridge_baseline(X, y),
        lambda: eval_ridge_tuned(X, y),
        lambda: eval_elasticnet_tuned(X, y),
        lambda: eval_lag_growth_plus_env(lag, X, y),
        lambda: eval_gbr_tuned(X, y),
        lambda: eval_rf_tuned(X, y),
        lambda: eval_arima_univariate(y),
        lambda: eval_sarimax(lag, X, y),
    ]
    results: list[dict[str, Any]] = []
    for run in runners:
        out = run()
        if out.get("skip"):
            continue
        results.append(out)
    return results


def _refit_best_pipeline(
    model_name: str,
    best_params: dict[str, Any],
    lag: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[Pipeline, list[str]]:
    """LOO CV로 고른 최적 모델을 전체 데이터에 재학습 (Optuna 예측용)."""
    if model_name == "RandomForest (CV tuned)":
        pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(random_state=42))])
        pipe.set_params(**best_params)
        pipe.fit(X, y)
        return pipe, list(X.columns)

    if model_name == "GradientBoosting (CV tuned)":
        pipe = Pipeline([("scaler", StandardScaler()), ("model", GradientBoostingRegressor(random_state=42))])
        pipe.set_params(**best_params)
        pipe.fit(X, y)
        return pipe, list(X.columns)

    if model_name in {"Ridge (CV tuned)", "ElasticNet (CV tuned)"}:
        model_cls = Ridge if model_name.startswith("Ridge") else ElasticNet
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model_cls(max_iter=5000))])
        pipe.set_params(**best_params)
        pipe.fit(X, y)
        return pipe, list(X.columns)

    if model_name == "Ridge + lag1 생장량 (CV tuned)":
        ext = X.copy()
        ext["prev_growth"] = lag["target"].shift(1)
        ext = ext.iloc[1:].reset_index(drop=True)
        y2 = y.iloc[1:].reset_index(drop=True)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
        pipe.set_params(**best_params)
        pipe.fit(ext, y2)
        return pipe, list(ext.columns)

    if model_name.startswith("Ridge (alpha="):
        alpha = float(best_params.get("alpha", 10.0))
        pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))])
        pipe.fit(X, y)
        return pipe, list(X.columns)

    # 시계열/베이스라인은 sklearn pipeline이 아니므로 RF로 대체
    pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(random_state=42))])
    rf = eval_rf_tuned(X, y)
    pipe.set_params(**rf["best_params"])
    pipe.fit(X, y)
    return pipe, list(X.columns)


def train_best_model_bundle() -> dict[str, Any]:
    """MAE 기준 최고 모델 선택 → 재학습 → Optuna 연동용 bundle 반환."""
    merged = load_merged()
    lag = build_lag_dataset(merged)
    X, y, _ = _prepare_xy(lag)
    results = _run_all_evaluations(lag, X, y)

    rows = [{k: v for k, v in r.items() if k != "pred"} for r in results]
    summary = pd.DataFrame(rows).sort_values("MAE_g").reset_index(drop=True)
    best_name = str(summary.iloc[0]["model"])
    best_out = next(r for r in results if r["model"] == best_name)

    pipe, feature_cols = _refit_best_pipeline(best_name, best_out["best_params"], lag, X, y)

    n_pred = len(best_out["pred"])
    lag_sub = lag.tail(n_pred).reset_index(drop=True)
    cv_df = pd.DataFrame(
        {
            "md": lag_sub["md"].values,
            "DAT": lag_sub["DAT"].values,
            "actual_g": lag_sub["target"].values,
            "predicted_g": best_out["pred"],
            "error_g": lag_sub["target"].values - best_out["pred"],
        }
    )
    metrics = {k: best_out[k] for k in ["MAE_g", "RMSE_g", "R2", "MAPE_pct", "n_samples"]}
    metrics["model"] = best_name

    bundle = ModelBundle(
        pipeline=pipe,
        feature_cols=feature_cols,
        train_df=lag,
        metrics=metrics,
        cv_predictions=cv_df,
    )

    summary.to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    cv_df.to_csv(OUT / "model_best_cv_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(OUT / "model_metrics.csv", index=False, encoding="utf-8-sig")

    return {
        "bundle": bundle,
        "lag_df": lag,
        "merged": merged,
        "summary": summary,
        "best_model": best_name,
        "best_params": best_out["best_params"],
    }


def compare_all_models() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_merged()
    lag = build_lag_dataset(df)
    X, y, _ = _prepare_xy(lag)
    results = _run_all_evaluations(lag, X, y)

    rows = [{k: v for k, v in r.items() if k != "pred"} for r in results]
    summary = pd.DataFrame(rows).sort_values("MAE_g").reset_index(drop=True)

    best_name = summary.iloc[0]["model"]
    best_out = next(r for r in results if r["model"] == best_name)

    pred_rows: list[dict[str, Any]] = []
    if best_out is not None:
        n_pred = len(best_out["pred"])
        lag_sub = lag.tail(n_pred).reset_index(drop=True)
        for i in range(n_pred):
            pred_rows.append(
                {
                    "md": lag_sub.loc[i, "md"],
                    "DAT": lag_sub.loc[i, "DAT"],
                    "actual_g": lag_sub.loc[i, "target"],
                    "predicted_g": best_out["pred"][i],
                    "model": best_name,
                }
            )

    pred_df = pd.DataFrame(pred_rows) if pred_rows else pd.DataFrame()
    summary.to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    if not pred_df.empty:
        pred_df.to_csv(OUT / "model_best_cv_predictions.csv", index=False, encoding="utf-8-sig")

    return summary, pred_df, lag


if __name__ == "__main__":
    summary, _, _ = compare_all_models()
    print("=== 모델 비교 (LOO CV, MAE 낮을수록 좋음) ===")
    cols = ["model", "MAE_g", "RMSE_g", "R2", "within_1g_pct", "within_2g_pct", "best_params"]
    print(summary[cols].to_string(index=False))
