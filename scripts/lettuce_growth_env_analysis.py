"""Lettuce growth vs environment change analysis."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

# 저장소 루트 = scripts/ 의 상위 폴더 (이식 가능한 경로)
ROOT = Path(__file__).resolve().parents[1]
# 원본 엑셀(4_Biomass.xlsx, 5_Environment_Solution.xlsx) 위치.
# 없으면 data/lettuce_processed 의 사전 생성 CSV를 그대로 사용합니다.
BASE = Path(os.environ.get("LETTUCE_RAW_DIR", ROOT / "data" / "lettuce_raw"))
OUT = ROOT / "data" / "lettuce_processed"
OUT.mkdir(parents=True, exist_ok=True)

# 원시 엑셀 컬럼 (상·하부 분리)
RAW_ENV_COLS = [
    "EC_dS_m",
    "WaterTemp_C",
    "AirTemp_Lower_C",
    "AirHum_Lower_pct",
    "AirTemp_Upper_C",
    "AirHum_Upper_pct",
    "Volume_L",
]
# 일별 집계에 사용할 통합 환경 변수 (상·하부 시간별 평균)
VALUE_COLS = [
    "EC_dS_m",
    "WaterTemp_C",
    "AirTemp_C",
    "AirHum_pct",
    "Volume_L",
]
SHEET_PLANTS = {
    "B2": ["B22", "B23", "B24", "B25", "B26", "B27"],
    "B3": ["B32", "B33", "B34", "B35", "B36", "B37"],
    "B4": ["B42", "B43", "B44", "B45", "B46", "B47"],
}
ROW_LABEL = {"B2": "lower", "B3": "middle", "B4": "upper"}


def build_hourly_env() -> pd.DataFrame:
    env_raw = pd.read_excel(BASE / "5_Environment_Solution.xlsx")
    env_raw.columns = ["Timestamp", *RAW_ENV_COLS]
    env_raw["Timestamp"] = pd.to_datetime(env_raw["Timestamp"])
    env_raw = env_raw.sort_values("Timestamp").set_index("Timestamp")
    env_h = env_raw.resample("1h").mean().reset_index()
    env_h["AirTemp_C"] = env_h[["AirTemp_Lower_C", "AirTemp_Upper_C"]].mean(axis=1)
    env_h["AirHum_pct"] = env_h[["AirHum_Lower_pct", "AirHum_Upper_pct"]].mean(axis=1)
    for col in VALUE_COLS:
        env_h[f"{col}_delta"] = env_h[col].diff()
        env_h[f"{col}_delta_pct"] = env_h[col].pct_change(fill_method=None) * 100
    env_h.to_csv(OUT / "environment_hourly.csv", index=False, encoding="utf-8-sig")
    return env_h


def build_biomass_long() -> pd.DataFrame:
    frames = []
    for sheet, plants in SHEET_PLANTS.items():
        df = pd.read_excel(BASE / "4_Biomass.xlsx", sheet_name=sheet)
        df["Date"] = pd.to_datetime(df["Date"])
        col_map = {
            c: p for c in df.columns for p in plants if p in str(c).replace(" ", "")
        }
        id_cols = [c for c in df.columns if c in col_map]
        long = df.melt(
            id_vars=["Date", "DAT (day)", "Biomass measured time"],
            value_vars=id_cols,
            var_name="plant_col",
            value_name="biomass_g",
        )
        long["plant_id"] = long["plant_col"].map(col_map)
        long["sheet"] = sheet
        long["row_tier"] = ROW_LABEL[sheet]
        frames.append(long.drop(columns=["plant_col"]))

    bio = pd.concat(frames, ignore_index=True).sort_values(["plant_id", "Date"])
    bio["DAT_numeric"] = pd.to_numeric(bio["DAT (day)"], errors="coerce")
    bio["biomass_delta_g"] = bio.groupby("plant_id")["biomass_g"].diff()
    bio["biomass_delta_pct"] = (
        bio.groupby("plant_id")["biomass_g"].pct_change(fill_method=None) * 100
    )
    bio.to_csv(OUT / "biomass_growth.csv", index=False, encoding="utf-8-sig")
    return bio


def daily_env_features(env_h: pd.DataFrame) -> pd.DataFrame:
    env_h = env_h.copy()
    env_h["md"] = env_h["Timestamp"].dt.strftime("%m-%d")
    delta_cols = [c for c in env_h.columns if c.endswith("_delta") and not c.endswith("_delta_pct")]

    agg_dict = {}
    for col in VALUE_COLS:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_min"] = (col, "min")
        agg_dict[f"{col}_max"] = (col, "max")
        agg_dict[f"{col}_range"] = (col, lambda s: s.max() - s.min())
        agg_dict[f"{col}_daily_net_change"] = (
            col,
            lambda s: s.iloc[-1] - s.iloc[0] if len(s) > 1 else np.nan,
        )
    for col in delta_cols:
        agg_dict[f"{col}_sum"] = (col, "sum")
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_abs_mean"] = (col, lambda s: s.abs().mean())

    return env_h.groupby("md").agg(**agg_dict).reset_index()


def daily_biomass_features(bio: pd.DataFrame) -> pd.DataFrame:
    bio = bio.copy()
    bio["md"] = bio["Date"].dt.strftime("%m-%d")
    return (
        bio.groupby("md")
        .agg(
            DAT=("DAT_numeric", "first"),
            biomass_g_mean=("biomass_g", "mean"),
            biomass_delta_g_mean=("biomass_delta_g", "mean"),
            biomass_delta_g_median=("biomass_delta_g", "median"),
            biomass_delta_g_sum=("biomass_delta_g", "sum"),
            n_plants=("biomass_delta_g", lambda s: s.notna().sum()),
        )
        .reset_index()
    )


def plant_level_merge(bio: pd.DataFrame, env_h: pd.DataFrame) -> pd.DataFrame:
    """Match each plant's growth interval to env on same calendar day (month-day)."""
    env_h = env_h.copy()
    env_h["md"] = env_h["Timestamp"].dt.strftime("%m-%d")
    daily_env = daily_env_features(env_h)

    bio = bio.copy()
    bio["md"] = bio["Date"].dt.strftime("%m-%d")
    return bio.merge(daily_env, on="md", how="inner")


def correlation_table(df: pd.DataFrame, target: str) -> pd.Series:
    skip = {
        "md",
        "Date",
        "DAT",
        "DAT_numeric",
        "plant_id",
        "sheet",
        "row_tier",
        "Biomass measured time",
        target,
        "biomass_g",
        "biomass_delta_pct",
        "biomass_g_mean",
        "biomass_delta_g_sum",
        "biomass_delta_g_median",
        "n_plants",
    }
    feature_cols = [c for c in df.columns if c not in skip and df[c].dtype != "object"]
    return (
        df[feature_cols + [target]]
        .corr(numeric_only=True)[target]
        .drop(target)
        .sort_values(ascending=False)
    )


def high_low_comparison(df: pd.DataFrame, target: str, top_features: int = 15) -> pd.DataFrame:
    valid = df.dropna(subset=[target])
    q75 = valid[target].quantile(0.75)
    q25 = valid[target].quantile(0.25)
    high = valid[valid[target] >= q75]
    low = valid[valid[target] <= q25]

    env_cols = [
        c
        for c in valid.columns
        if any(
            c.startswith(f"{v}_")
            for v in VALUE_COLS
        )
        and valid[c].dtype != "object"
    ]
    diff = high[env_cols].mean() - low[env_cols].mean()
    out = pd.DataFrame(
        {
            "feature": diff.index,
            "high_minus_low": diff.values,
            "high_day_mean": high[env_cols].mean().values,
            "low_day_mean": low[env_cols].mean().values,
        }
    ).sort_values("high_minus_low", ascending=False)
    return out.head(top_features), out.tail(top_features), q75, q25


def main() -> None:
    env_h = build_hourly_env()
    bio = build_biomass_long()

    daily_bio = daily_biomass_features(bio)
    daily_env = daily_env_features(env_h)
    merged_daily = daily_bio.merge(daily_env, on="md", how="inner")
    merged_daily.to_csv(OUT / "daily_merged_bio_env.csv", index=False, encoding="utf-8-sig")

    plant_merged = plant_level_merge(bio, env_h)
    plant_merged.to_csv(OUT / "plant_daily_merged_bio_env.csv", index=False, encoding="utf-8-sig")

    target = "biomass_delta_g_mean"
    daily_corrs = correlation_table(merged_daily, target)
    plant_corrs = correlation_table(plant_merged, "biomass_delta_g")

    daily_corrs.to_csv(OUT / "growth_env_correlation_daily.csv", encoding="utf-8-sig")
    plant_corrs.to_csv(OUT / "growth_env_correlation_plant.csv", encoding="utf-8-sig")

    top_pos, top_neg, q75, q25 = high_low_comparison(merged_daily, target)
    compare = pd.concat(
        [
            top_pos.assign(group="high_growth_favors"),
            top_neg.assign(group="low_growth_favors"),
        ]
    )
    compare.to_csv(OUT / "high_vs_low_growth_env_diff.csv", index=False, encoding="utf-8-sig")

    # lag-1: previous day env vs today's growth
    lag_df = merged_daily.sort_values("DAT").copy()
    env_feature_cols = [c for c in lag_df.columns if c.startswith("EC_") or c.startswith("Water") or c.startswith("Air") or c.startswith("Volume")]
    env_feature_cols = [c for c in env_feature_cols if lag_df[c].dtype != "object"]
    for col in env_feature_cols:
        lag_df[f"prev_{col}"] = lag_df[col].shift(1)
    lag_cols = [c for c in lag_df.columns if c.startswith("prev_")]
    lag_corrs = (
        lag_df[lag_cols + [target]]
        .corr(numeric_only=True)[target]
        .drop(target)
        .sort_values(ascending=False)
    )
    lag_corrs.to_csv(OUT / "growth_env_correlation_lag1.csv", encoding="utf-8-sig")

    # actionable summary: top drivers
    summary_rows = []
    for name, series in [
        ("same_day", daily_corrs),
        ("plant_level", plant_corrs),
        ("lag1_prev_day", lag_corrs),
    ]:
        summary_rows.append(
            {
                "analysis": name,
                "top_positive_feature": series.index[0],
                "top_positive_r": round(series.iloc[0], 4),
                "top_negative_feature": series.index[-1],
                "top_negative_r": round(series.iloc[-1], 4),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "growth_env_summary.csv", index=False, encoding="utf-8-sig")

    print("=== DAILY GROWTH vs ENV (top +) ===")
    print(daily_corrs.head(8).to_string())
    print("\n=== DAILY GROWTH vs ENV (top -) ===")
    print(daily_corrs.tail(8).to_string())
    print(f"\n=== HIGH (>= {q75:.2f} g/day) vs LOW (<= {q25:.2f} g/day) ===")
    print(top_pos[["feature", "high_minus_low"]].to_string(index=False))
    print("\n=== LAG-1 (prev day) top + ===")
    print(lag_corrs.head(6).to_string())
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
