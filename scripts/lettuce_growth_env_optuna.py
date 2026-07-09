"""Optuna 환경 최적화 - 전날 환경 조정 시 예상 생장량(g/주/일).

model_develop_0626 / lettuce_growth_model 파이프라인과 동일:
  prev_* 환경 → Ridge 예측 → biomass_delta_g_mean (18주 평균 일일 생장)
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from lettuce_growth_model import (
    KEY_FEATURES,
    ModelBundle,
    build_lag_dataset,
    load_merged,
    train_growth_model,
)

warnings.filterwarnings("ignore")

OUT = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "lettuce_processed"

# Optuna가 직접 맞출 수 있는 환경 setpoint (Volume·요동 제외)
DEFAULT_CONTROL = [
    "AirHum_pct_min",
    "AirTemp_C_min",
    "WaterTemp_C_min",
    "EC_dS_m_mean",
]

LABEL_KO = {
    "AirHum_pct_min": "습도 최저(%)",
    "AirTemp_C_min": "기온 최저(°C)",
    "WaterTemp_C_min": "수온 최저(°C)",
    "EC_dS_m_mean": "EC 평균(dS/m)",
    "Volume_L_mean": "양액량 평균(L)",
    "AirTemp_C_delta_abs_mean": "기온 요동(°C/h)",
    "AirHum_pct_daily_net_change": "습도 일일 순변화",
    "Volume_L_daily_net_change": "양액량 일일 순변화",
}

UNIT_MAP = {
    "AirHum_pct_min": "%",
    "AirTemp_C_min": "°C",
    "WaterTemp_C_min": "°C",
    "EC_dS_m_mean": "dS/m",
    "Volume_L_mean": "L",
    "AirTemp_C_delta_abs_mean": "°C/h",
    "AirHum_pct_daily_net_change": "%/일",
    "Volume_L_daily_net_change": "L/일",
}

MAX_STEP = {
    "AirHum_pct_min": 3.0,
    "AirTemp_C_min": 0.5,
    "WaterTemp_C_min": 0.5,
    "EC_dS_m_mean": 0.10,
    "Volume_L_mean": 5.0,
    "AirTemp_C_delta_abs_mean": 0.15,
    "AirHum_pct_daily_net_change": 2.0,
    "Volume_L_daily_net_change": 2.0,
}


def _prev_col(feat: str) -> str:
    return f"prev_{feat}"


def _build_bounds(lag_df: pd.DataFrame, control_feats: list[str]) -> dict[str, dict[str, float]]:
    bounds = {}
    for feat in control_feats:
        col = _prev_col(feat)
        series = pd.to_numeric(lag_df[col], errors="coerce").dropna()
        low = float(series.quantile(0.10))
        high = float(series.quantile(0.90))
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            low, high = float(series.min()), float(series.max())
        if low >= high:
            pad = max(abs(low) * 0.05, 0.1)
            low, high = low - pad, high + pad
        bounds[feat] = {"low": low, "high": high}
    return bounds


def _resolve_day_index(
    merged: pd.DataFrame,
    day_index: int | None = None,
    md: str | None = None,
    dat: float | int | None = None,
) -> int:
    """기준일 인덱스: md / DAT / day_index 중 하나로 지정 (기본=마지막 날)."""
    if md is not None:
        hits = merged.index[merged["md"].astype(str) == str(md)].tolist()
        if not hits:
            raise ValueError(f"md='{md}' 에 해당하는 날짜가 없습니다.")
        return int(hits[0])
    if dat is not None:
        hits = merged.index[merged["DAT"] == float(dat)].tolist()
        if not hits:
            raise ValueError(f"DAT={dat} 에 해당하는 날짜가 없습니다.")
        return int(hits[0])
    if day_index is None:
        return len(merged) - 1
    if day_index < 0:
        return len(merged) + day_index
    return int(day_index)


def _base_feature_row(
    lag_df: pd.DataFrame,
    merged: pd.DataFrame,
    day_index: int | None = None,
    md: str | None = None,
    dat: float | int | None = None,
) -> tuple[pd.Series, int]:
    """지정한 관측일 환경을 prev_* 벡터로 구성."""
    idx = _resolve_day_index(merged, day_index=day_index, md=md, dat=dat)
    last = merged.iloc[idx]
    row = lag_df.iloc[min(idx, len(lag_df) - 1)].copy()
    for feat in KEY_FEATURES:
        col = _prev_col(feat)
        if feat in last.index:
            row[col] = last[feat]
    return row, idx


def _predict_growth(pipe: Pipeline, feature_cols: list[str], row: pd.Series) -> float:
    x = pd.DataFrame([row[feature_cols].astype(float)])
    return float(pipe.predict(x)[0])


def _search_random(
    pipe: Pipeline,
    feature_cols: list[str],
    base_row: pd.Series,
    control_feats: list[str],
    bounds: dict,
    n_samples: int = 300,
) -> tuple[dict[str, float], float]:
    rng = np.random.default_rng(42)
    best_score = -np.inf
    best_params: dict[str, float] = {}

    for _ in range(n_samples):
        trial = base_row.copy()
        params = {}
        for feat in control_feats:
            v = float(rng.uniform(bounds[feat]["low"], bounds[feat]["high"]))
            trial[_prev_col(feat)] = v
            params[feat] = v
        try:
            score = _predict_growth(pipe, feature_cols, trial)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_params = params

    if not best_params:
        best_params = {
            f: float(base_row[_prev_col(f)])
            for f in control_feats
            if _prev_col(f) in base_row.index
        }
        best_score = _predict_growth(pipe, feature_cols, base_row)

    return best_params, best_score


def _build_control_df(
    control_feats: list[str],
    current_env: dict[str, float],
    best_env: dict[str, float],
    bounds: dict,
) -> pd.DataFrame:
    rows = []
    for feat in control_feats:
        cur = float(current_env[feat])
        tgt = float(best_env.get(feat, cur))
        diff = tgt - cur
        span = bounds[feat]["high"] - bounds[feat]["low"]
        threshold = span * 0.03
        step = MAX_STEP.get(feat, abs(diff))

        if diff > threshold:
            direction, advice = "increase", "높이는 방향"
            nxt = cur + min(abs(diff), step)
        elif diff < -threshold:
            direction, advice = "decrease", "낮추는 방향"
            nxt = cur - min(abs(diff), step)
        else:
            direction, advice = "maintain", "현재 수준 유지"
            nxt = cur

        rows.append({
            "feature": feat,
            "variable": LABEL_KO.get(feat, feat),
            "unit": UNIT_MAP.get(feat, ""),
            "current_value": round(cur, 3),
            "bayesian_target": round(tgt, 3),
            "next_control_value": round(nxt, 3),
            "difference": round(diff, 3),
            "direction": direction,
            "advice": advice,
        })

    df = pd.DataFrame(rows)
    df["target_adjustment"] = df["bayesian_target"] - df["current_value"]
    df["next_adjustment"] = df["next_control_value"] - df["current_value"]
    df["target_adjustment_%"] = (
        df["target_adjustment"] / df["current_value"].replace(0, np.nan) * 100
    ).round(2)
    df["next_adjustment_%"] = (
        df["next_adjustment"] / df["current_value"].replace(0, np.nan) * 100
    ).round(2)

    def _summary(row):
        v, u = row["variable"], row["unit"]
        cur, nxt, tgt = row["current_value"], row["next_control_value"], row["bayesian_target"]
        if row["direction"] == "maintain":
            return f"{v}: 현재 {cur}{u} 유지"
        sign_n = "+" if row["next_adjustment"] >= 0 else ""
        sign_t = "+" if row["target_adjustment"] >= 0 else ""
        return (
            f"{v}: 현재 {cur}{u} → 다음 제어 {nxt}{u} ({sign_n}{row['next_adjustment']:.3f}{u}), "
            f"장기 목표 {tgt}{u} ({sign_t}{row['target_adjustment']:.3f}{u})"
        )

    df["adjustment_summary"] = df.apply(_summary, axis=1)
    return df


def run_env_control_optuna(
    bundle: ModelBundle | None = None,
    lag_df: pd.DataFrame | None = None,
    merged: pd.DataFrame | None = None,
    n_trials: int = 150,
    alpha: float = 10.0,
    control_features: list[str] | None = None,
    day_index: int | None = None,
    base_md: str | None = None,
    base_dat: float | int | None = None,
    env_override: dict[str, float] | None = None,
    scenario_name: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    환경 setpoint를 조정했을 때 생장 모델이 예측하는 내일 생장량(g/주/일) 변화를 계산합니다.

    Parameters
    ----------
    day_index, base_md, base_dat
        기준일 (미지정 시 마지막 관측일). 예: base_md='03-10'
    env_override
        기준 환경 수동 변경. 예: {'AirHum_pct_min': 35, 'WaterTemp_C_min': 19}
    scenario_name
        출력·결과에 붙일 시나리오 이름
    verbose
        False면 print 생략 (여러 시나리오 일괄 실행용)
    """
    if merged is None:
        merged = load_merged()
    if lag_df is None or bundle is None:
        from lettuce_growth_model_compare import train_best_model_bundle

        best = train_best_model_bundle()
        bundle = best["bundle"]
        lag_df = best["lag_df"]
        merged = best["merged"]

    control_feats = control_features or [f for f in DEFAULT_CONTROL if f in KEY_FEATURES]
    feature_cols = list(bundle.feature_cols)
    pipe = bundle.pipeline

    bounds = _build_bounds(lag_df, control_feats)
    base_row, base_idx = _base_feature_row(
        lag_df, merged, day_index=day_index, md=base_md, dat=base_dat
    )

    if env_override:
        for feat, val in env_override.items():
            if feat in KEY_FEATURES:
                base_row[_prev_col(feat)] = float(val)

    current_env = {
        f: float(base_row[_prev_col(f)])
        for f in control_feats
        if _prev_col(f) in base_row.index
    }

    ref = merged.iloc[base_idx]
    latest_md = str(ref.get("md", ""))
    latest_dat = ref.get("DAT", "")
    label = scenario_name or f"{latest_md} (DAT {latest_dat})"

    optuna_success = False
    search_method = "random_search"
    best_env: dict[str, float] = {}
    best_pred = np.nan

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            row = base_row.copy()
            for feat in control_feats:
                row[_prev_col(feat)] = trial.suggest_float(
                    feat, bounds[feat]["low"], bounds[feat]["high"]
                )
            return _predict_growth(pipe, feature_cols, row)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_env = study.best_params
        best_pred = float(study.best_value)
        optuna_success = True
        search_method = "optuna"
    except ImportError:
        print("optuna 미설치 → 랜덤 탐색으로 대체 (pip install optuna 권장)")
    except Exception as exc:
        print(f"Optuna 실패 → 랜덤 탐색: {exc}")

    if not optuna_success:
        best_env, best_pred = _search_random(
            pipe, feature_cols, base_row, control_feats, bounds,
            n_samples=max(200, n_trials * 2),
        )

    control_df = _build_control_df(control_feats, current_env, best_env, bounds)

    def _scenario_row(name: str, env_overrides: dict[str, float]) -> tuple[str, float]:
        row = base_row.copy()
        for feat, val in env_overrides.items():
            row[_prev_col(feat)] = val
        pred = _predict_growth(pipe, feature_cols, row)
        return name, pred

    current_pred = _scenario_row("current_environment", current_env)[1]
    target_pred = _scenario_row("bayesian_target", best_env)[1]
    next_env = {
        f: float(control_df.loc[control_df["feature"] == f, "next_control_value"].iloc[0])
        for f in control_feats
    }
    next_pred = _scenario_row("next_control_value", next_env)[1]

    def _improve(name, pred, base):
        return {
            "scenario": name,
            "pred_growth_g_per_day": round(pred, 3),
            "growth_increase_g": round(pred - base, 3),
            "growth_increase_%": round((pred - base) / abs(base) * 100 if base else 0, 2),
        }

    improvement_df = pd.DataFrame([
        _improve("current_environment", current_pred, current_pred),
        _improve("bayesian_target", target_pred, current_pred),
        _improve("next_control_value", next_pred, current_pred),
    ])

    if verbose:
        print("\n" + "=" * 60)
        print("Optuna 환경 최적화 - 예상 일일 생장 (18주 평균, g/주/일)")
        print("=" * 60)
        model_name = bundle.metrics.get("model", "생장 예측 모델")
        print(f"시나리오: {label} | 모델: {model_name}")
        if env_override:
            print(f"  ※ 수동 환경 변경: {env_override}")
        print(f"탐색: {search_method} ({n_trials} trials) | 최대 예측 생장: {best_pred:.3f} g/일")
        print("\n[환경 조정 권고]")
        print(control_df[["variable", "current_value", "next_control_value", "bayesian_target", "advice"]].to_string(index=False))
        print("\n[조정 요약]")
        for line in control_df["adjustment_summary"]:
            print(" ", line)
        print("\n[시나리오별 예상 생장]")
        print(improvement_df.to_string(index=False))
        print("\n[해석]")
        print(
            f"현재 환경 기준 예측 생장 {current_pred:.2f} g/일 → "
            f"장기 목표 환경 {target_pred:.2f} g/일 "
            f"(+{target_pred - current_pred:.2f} g, "
            f"{(target_pred - current_pred) / abs(current_pred) * 100:.1f}% )."
        )
        print(
            f"다음 1회 제어(step 제한) 적용 시: {next_pred:.2f} g/일 "
            f"(+{next_pred - current_pred:.2f} g)."
        )
        print("※ 상관·회귀 기반 참고값이며, 인과 보장은 없습니다.")

    out_suffix = ""
    if scenario_name:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in scenario_name)
        out_suffix = f"_{safe}"

    control_df.to_csv(OUT / f"env_control_optuna{out_suffix}.csv", index=False, encoding="utf-8-sig")
    improvement_df.to_csv(OUT / f"env_control_improvement{out_suffix}.csv", index=False, encoding="utf-8-sig")

    return {
        "scenario_name": label,
        "base_idx": base_idx,
        "base_md": latest_md,
        "base_dat": latest_dat,
        "env_override": env_override or {},
        "optuna_success": optuna_success,
        "search_method": search_method,
        "control_df": control_df,
        "improvement_df": improvement_df,
        "best_env": best_env,
        "best_pred_growth_g": best_pred,
        "current_env": current_env,
        "current_pred_g": current_pred,
        "target_pred_g": target_pred,
        "next_pred_g": next_pred,
        "latest_md": latest_md,
        "latest_dat": latest_dat,
        "bundle": bundle,
        "bounds": bounds,
    }


def run_env_scenarios(
    scenarios: list[dict[str, Any]],
    bundle: ModelBundle | None = None,
    lag_df: pd.DataFrame | None = None,
    merged: pd.DataFrame | None = None,
    n_trials: int = 100,
    verbose_each: bool = False,
) -> pd.DataFrame:
    """여러 환경 시나리오를 순서대로 Optuna 실행 후 요약표 반환."""
    rows = []
    for spec in scenarios:
        name = spec.get("name", "시나리오")
        kwargs = {k: v for k, v in spec.items() if k != "name"}
        res = run_env_control_optuna(
            bundle=bundle,
            lag_df=lag_df,
            merged=merged,
            n_trials=n_trials,
            scenario_name=name,
            verbose=verbose_each,
            **kwargs,
        )
        cur = res["current_env"]
        rows.append({
            "시나리오": name,
            "기준일": res["base_md"],
            "DAT": res["base_dat"],
            "습도(%)": round(cur.get("AirHum_pct_min", np.nan), 2),
            "기온(°C)": round(cur.get("AirTemp_C_min", np.nan), 2),
            "수온(°C)": round(cur.get("WaterTemp_C_min", np.nan), 2),
            "EC": round(cur.get("EC_dS_m_mean", np.nan), 3),
            "현재 예측(g/일)": round(res["current_pred_g"], 3),
            "다음 제어(g/일)": round(res["next_pred_g"], 3),
            "장기 목표(g/일)": round(res["target_pred_g"], 3),
            "증가(g)": round(res["target_pred_g"] - res["current_pred_g"], 3),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "env_scenario_comparison.csv", index=False, encoding="utf-8-sig")
    return summary


if __name__ == "__main__":
    run_env_control_optuna(n_trials=100)
