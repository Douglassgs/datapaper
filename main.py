"""全球碳排放与能源结构时空可视化分析"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ==================== 全局配置 ====================
matplotlib.rcParams["font.sans-serif"] = [ "SimHei","Microsoft JhengHei UI", "Noto Sans SC"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["savefig.bbox"] = "tight"

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 12 个代表性国家 + World
SELECTED_COUNTRIES = [
    "China", "United States", "India", "Russia", "Japan",
    "Germany", "United Kingdom", "France", "Brazil",
    "Saudi Arabia", "Iran", "South Korea", "World",
]

# 能源源类别
ENERGY_SOURCES = ["coal_co2", "oil_co2", "gas_co2", "cement_co2", "other_industry_co2"]

# 基础字段
BASE_FIELDS = [
    "country", "year", "iso_code", "population", "gdp",
    "co2", "co2_per_capita", "co2_per_gdp", "co2_growth_prct",
    "co2_including_luc", "primary_energy_consumption",
    "energy_per_capita", "energy_per_gdp",
    *ENERGY_SOURCES,
]

# ==================== 数据加载 ====================
def load_owid_data(path: str) -> pd.DataFrame:
    """加载 OWID二氧化碳数据集并筛选 1990 年以后的年份"""
    df = pd.read_csv(path)
    df = df[df["year"] >= 1990].copy()
    return df


def classify_locations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """将数据分为国家实体、区域汇总、世界总计三类"""
    regional = df[df["iso_code"].isna()].copy()
    countries = df[df["iso_code"].notna()].copy()
    world = df[df["country"] == "World"].copy()
    return countries, regional, world


# ==================== 数据清洗 ====================
def check_duplicates(df: pd.DataFrame) -> int:
    """检查 location-year 主键重复"""
    dupes = df.duplicated(subset=["country", "year"], keep=False)
    return dupes.sum()


def missing_report(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """生成字段缺失率报告"""
    report = []
    for col in fields:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            n_total = len(df)
            report.append({"字段": col, "缺失数": n_miss, "缺失率(%)": round(n_miss / n_total * 100, 2)})
    return pd.DataFrame(report)


def detect_outliers(df: pd.DataFrame, col: str) -> pd.Series:
    """IQR 方法检测异常值"""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    return (df[col] < lower) | (df[col] > upper)


def clean_country_data(df: pd.DataFrame) -> pd.DataFrame:
    """对选定国家的关键字段进行缺失值填充"""
    df = df.copy()
    key_fields = ["co2", "co2_per_capita", "population", "gdp", "primary_energy_consumption"]
    available = [f for f in key_fields if f in df.columns]
    for country in df["country"].unique():
        mask = df["country"] == country
        subset = df.loc[mask, ["year"] + available].sort_values("year")
        df.loc[mask, available] = subset[available].ffill().bfill().values
    return df


# ==================== 特征构建 ====================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """构建分析所需的派生特征"""
    df = df.copy()

    # 能源结构占比
    for src in ENERGY_SOURCES:
        df[f"{src}_share"] = df[src] / df["co2"].replace(0, np.nan) * 100

    # 5 年移动平均
    for country in df["country"].unique():
        mask = df["country"] == country
        df.loc[mask, "co2_ma5"] = (
            df.loc[mask].sort_values("year")["co2"].rolling(5, min_periods=1).mean().values
        )

    # GDP 年增速
    df["gdp_growth"] = df.groupby("country")["gdp"].pct_change() * 100

    # 脱钩指数：GDP 增速 -二氧化碳增速（正值表示脱钩）
    df["decoupling_index"] = df["gdp_growth"] - df["co2_growth_prct"]

    # 人均排放分组标签
    bins = [0, 2, 5, 10, float("inf")]
    labels = ["低排放(<2t)", "中排放(2-5t)", "高排放(5-10t)", "超高排放(>10t)"]
    df["emission_category"] = pd.cut(df["co2_per_capita"], bins=bins, labels=labels)

    return df


def compute_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """计算每年的二氧化碳排放排名"""
    rankings = df[df["iso_code"].notna()].copy()
    rankings["rank"] = rankings.groupby("year")["co2"].rank(ascending=False, method="min")
    return rankings[["country", "year", "co2", "rank"]]


# ==================== 质量报告 ====================
def print_quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame, outliers: dict):
    """输出数据质量检查报告"""
    print("=" * 60)
    print("数据质量检查报告")
    print("=" * 60)
    print(f"清洗前数据规模: {raw.shape[0]} 行 × {raw.shape[1]} 列")
    print(f"清洗后数据规模: {cleaned.shape[0]} 行 × {cleaned.shape[1]} 列")
    print(f"国家实体数: {cleaned['country'].nunique()}")
    print(f"年份范围: {cleaned['year'].min()} — {cleaned['year'].max()}")
    print()

    miss = missing_report(raw, BASE_FIELDS)
    print("关键字段缺失率（整体）:")
    print(miss.to_string(index=False))
    print()

    if outliers:
        print("异常值检测结果（IQR 3倍范围）:")
        for col, count in outliers.items():
            print(f"  {col}: {count} 条异常记录")
    print()


# ==================== 图表绘制 ====================
def mil_formatter(x, pos=None):
    """百万格式化器"""
    return f"{x/1e3:.0f}" if x < 1e4 else f"{x/1e6:.1f}Gt"


def fig1_co2_trend(df: pd.DataFrame):
    """图1 - 全球及重点国家二氧化碳排放总量时序图"""
    country_df = df[df["country"].isin(SELECTED_COUNTRIES)]

    # 按 2024 年排放量排序（除去 World）
    latest = country_df[country_df["year"] == country_df["year"].max()]
    order = (
        latest[latest["country"] != "World"]
        .sort_values("co2", ascending=False)["country"]
        .tolist()
    )
    colors = plt.cm.tab20(np.linspace(0, 1, len(order)))

    # 断轴：跳过全球排放与其他国家之间的空白区间
    non_world = country_df[country_df["country"] != "World"]
    world = country_df[country_df["country"] == "World"].sort_values("year")
    max_non_world = non_world["co2"].max()
    min_world = world["co2"].min()

    lower_max = max_non_world * 1.1
    upper_min = min_world * 0.9

    if lower_max < upper_min:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(14, 7),
            gridspec_kw={"height_ratios": [1, 2], "hspace": 0.05},
        )

        for i, country in enumerate(order):
            cdf = country_df[country_df["country"] == country].sort_values("year")
            ax_top.plot(cdf["year"], cdf["co2"], color=colors[i], linewidth=1.5, label=country)
            ax_bottom.plot(cdf["year"], cdf["co2"], color=colors[i], linewidth=1.5, label=country)

        ax_top.plot(world["year"], world["co2"], color="black", linewidth=2.5, linestyle="--", label="全球")
        ax_bottom.plot(world["year"], world["co2"], color="black", linewidth=2.5, linestyle="--", label="全球")

        ax_bottom.set_ylim(0, lower_max)
        ax_top.set_ylim(upper_min, world["co2"].max() * 1.05)

        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.tick_params(labeltop=False)

        # 断轴标记
        d = 0.008
        kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs = dict(transform=ax_bottom.transAxes, color="k", clip_on=False, linewidth=1)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ax_bottom.set_xlabel("年份")
        ax_bottom.set_ylabel("二氧化碳 排放量（百万吨/Mt）")
        ax_top.set_title("全球及重点国家二氧化碳排放总量变化（1990—2024）")
        ax_bottom.legend(loc="upper left", fontsize=8, ncol=2)
        ax_top.grid(True, alpha=0.3)
        ax_bottom.grid(True, alpha=0.3)
        fig.savefig(OUTPUT_DIR / "fig1_co2_trend.svg")
        plt.close(fig)
        print("图1 已保存")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, country in enumerate(order):
        cdf = country_df[country_df["country"] == country].sort_values("year")
        ax.plot(cdf["year"], cdf["co2"], color=colors[i], linewidth=1.5, label=country)

    # World 用粗黑线
    ax.plot(world["year"], world["co2"], color="black", linewidth=2.5, linestyle="--", label="全球")

    ax.set_xlabel("年份")
    ax.set_ylabel("二氧化碳 排放量（百万吨/Mt）")
    ax.set_title("全球及重点国家二氧化碳排放总量变化（1990—2024）")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(OUTPUT_DIR / "fig1_co2_trend.svg")
    plt.close(fig)
    print("图1 已保存")


def fig2_co2_per_capita(df: pd.DataFrame):
    """图2 - 重点国家二氧化碳人均排放时序图"""
    fig, ax = plt.subplots(figsize=(14, 7))
    country_df = df[df["country"].isin(SELECTED_COUNTRIES)]

    # 按最新年份人均排放排序（除去 World）
    latest = country_df[country_df["year"] == country_df["year"].max()]
    order = (
        latest[latest["country"] != "World"]
        .sort_values("co2_per_capita", ascending=False)["country"]
        .tolist()
    )
    colors = plt.cm.tab20(np.linspace(0, 1, len(order)))

    for i, country in enumerate(order):
        cdf = country_df[country_df["country"] == country].sort_values("year")
        ax.plot(cdf["year"], cdf["co2_per_capita"], color=colors[i], linewidth=1.5, label=country)

    world = country_df[country_df["country"] == "World"].sort_values("year")
    ax.plot(
        world["year"], world["co2_per_capita"],
        color="black", linewidth=2.5, linestyle="--", label="全球均值",
    )

    ax.set_xlabel("年份")
    ax.set_ylabel("人均二氧化碳排放（吨/人）")
    ax.set_title("重点国家人均二氧化碳排放变化（1990—2024）")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(OUTPUT_DIR / "fig2_co2_per_capita.svg")
    plt.close(fig)
    print("图2 已保存")


def fig3_energy_source(df: pd.DataFrame):
    """图3 - 全球二氧化碳排放源构成变化（堆叠面积图）"""
    world = df[df["country"] == "World"].sort_values("year")
    source_labels = {
        "coal_co2": "煤炭",
        "oil_co2": "石油",
        "gas_co2": "天然气",
        "cement_co2": "水泥",
        "other_industry_co2": "其他工业",
    }
    colors_map = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(14, 7))
    y_data = {cn: world[s].fillna(0).values for s, cn in source_labels.items()}
    ax.stackplot(world["year"].values, *y_data.values(), labels=source_labels.values(), colors=colors_map, alpha=0.85)

    ax.set_xlabel("年份")
    ax.set_ylabel("二氧化碳 排放量（百万吨/Mt）")
    ax.set_title("全球二氧化碳排放源构成变化（1990—2024）")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(OUTPUT_DIR / "fig3_energy_source.svg")
    plt.close(fig)
    print("图3 已保存")


def fig4_gdp_co2_scatter(df: pd.DataFrame):
    """图4 - GDP 与二氧化碳排放关系（plotly 交互气泡图）"""
    required_fields = ["gdp", "co2_per_capita", "population", "co2_per_gdp", "primary_energy_consumption"]
    country_df = df[df["iso_code"].notna()].copy()
    valid = country_df.dropna(subset=required_fields)

    # 使用最新“字段齐全”的年份，避免 2024 年 GDP 等字段缺失导致无数据
    valid = valid[(valid["gdp"] > 0) & (valid["co2_per_capita"] > 0)]
    if valid.empty:
        print("图4 无可用数据（关键字段为空或无正值）")
        return

    year = int(valid["year"].max())
    plot_df = valid[valid["year"] == year].copy()

    fig = px.scatter(
        plot_df,
        x="gdp",
        y="co2_per_capita",
        size="population",
        color="co2_per_gdp",
        hover_name="country",
        log_x=True,
        log_y=True,
        size_max=60,
        color_continuous_scale="RdYlGn_r",
        title=f"图4 GDP 与人均二氧化碳排放关系（{int(year)} 年）",
        labels={
            "gdp": "GDP（2011不变价国际$）",
            "co2_per_capita": "人均二氧化碳排放（吨/人）",
            "co2_per_gdp": "碳排放强度（kg/$）",
        },
    )
    fig.write_html(OUTPUT_DIR / "fig4_gdp_co2_scatter.html")
    print("图4 已保存")


def fig5_world_map(df: pd.DataFrame):
    """图5 - 全球人均二氧化碳排放空间分布（plotly choropleth）"""
    year = 2023
    map_df = df[(df["year"] == year) & df["iso_code"].notna()].copy()

    fig = px.choropleth(
        map_df,
        locations="iso_code",
        color="co2_per_capita",
        hover_name="country",
        color_continuous_scale="YlOrRd",
        range_color=(0, 20),
        title=f"图5 全球人均二氧化碳排放空间分布（{year} 年）",
        labels={
            "co2_per_capita": "人均 二氧化碳（吨/人）",
            "iso_code": "ISO 代码",
        },
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.write_html(OUTPUT_DIR / "fig5_world_map.html")
    print("图5 已保存")


def fig6_ranking(df: pd.DataFrame):
    """图6 - Top15 排放国排名变化（横向条形图）"""
    rankings = compute_rankings(df)
    snapshot_years = [1990, 2000, 2010, 2020, 2024]

    fig, axes = plt.subplots(1, 5, figsize=(22, 10), sharex=True)
    for i, yr in enumerate(snapshot_years):
        ax = axes[i]
        top15 = (
            rankings[rankings["year"] == yr]
            .dropna(subset=["co2"])
            .nlargest(15, "co2")
            .sort_values("co2", ascending=True)
        )
        bars = ax.barh(top15["country"], top15["co2"], color=plt.cm.viridis(np.linspace(0.2, 0.9, 15)))
        ax.set_title(f"{yr} 年", fontsize=12)
        for bar, val in zip(bars, top15["co2"]):
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}", va="center", fontsize=7)

    fig.supxlabel("二氧化碳 排放量（百万吨/Mt）")
    fig.suptitle("Top15 排放国排名变化（1990—2024）", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_ranking.svg")
    plt.close(fig)
    print("图6 已保存")


def fig7_energy_structure(df: pd.DataFrame):
    """图7 - 能源结构对比（堆叠柱状图）"""
    year = df["year"].max()
    country_df = df[(df["country"].isin(SELECTED_COUNTRIES)) & (df["country"] != "World") & (df["year"] == year)]

    source_labels = {"coal_co2_share": "煤炭", "oil_co2_share": "石油",
                     "gas_co2_share": "天然气", "cement_co2_share": "水泥",
                     "other_industry_co2_share": "其他工业"}
    colors_map = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    plot_data = country_df.sort_values("co2", ascending=True)
    x = plot_data["country"].tolist()
    bottom = np.zeros(len(x))

    fig, ax = plt.subplots(figsize=(12, 7))
    for (col, label), color in zip(source_labels.items(), colors_map):
        vals = plot_data[col].fillna(0).values
        ax.barh(x, vals, left=bottom, label=label, color=color, alpha=0.85)
        bottom += vals

    ax.set_xlabel("排放占比（%）")
    ax.set_title(f"重点国家能源结构对比（{int(year)} 年）")
    ax.legend(loc="lower right", fontsize=9, ncol=3)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_energy_structure.svg")
    plt.close(fig)
    print("图7 已保存")


# ==================== 补充数据：World Bank 气候风险 ====================
def load_worldbank_data(path: str) -> dict:
    """加载 World Bank 气候风险数据集中的人口相关 sheet"""
    xl = pd.ExcelFile(path)
    sheets = {}

    # 历史人口（2000 年）
    if "popcount_2000-2000_historical_g" in xl.sheet_names:
        raw = pd.read_excel(xl, "popcount_2000-2000_historical_g")
        sheets["hist_pop"] = raw.rename(columns={"2000-07": "population_2000"})

    # SSP 人口预测（选取 SSP1, SSP2, SSP3, SSP5）
    ssp_sheets = {
        "SSP1": "popcount_2010-2100_ssp119_gpw-v",
        "SSP2": "popcount_2010-2100_ssp245_gpw-v",
        "SSP3": "popcount_2010-2100_ssp370_gpw-v",
        "SSP5": "popcount_2010-2100_ssp585_gpw-v",
    }
    for label, sname in ssp_sheets.items():
        if sname in xl.sheet_names:
            raw = pd.read_excel(xl, sname)
            raw["ssp"] = label
            # 将年份列熔化为长表
            yr_cols = [c for c in raw.columns if c not in ("code", "name", "ssp")]
            melted = raw.melt(
                id_vars=["code", "name", "ssp"], value_vars=yr_cols,
                var_name="year_str", value_name="population",
            )
            melted["year"] = melted["year_str"].str.extract(r"(\d{4})").astype(int)
            sheets[f"ssp_{label}"] = melted.drop(columns=["year_str"])

    xl.close()
    return sheets


def fig8_ssp_population(wb_data: dict, owid_df: pd.DataFrame | None = None):
    """图8 - SSP 情景下人口预测对比"""
    target_countries = ["China", "India", "United States", "Nigeria"]
    ssp_order = ["SSP1", "SSP2", "SSP3", "SSP5"]
    colors = {"SSP1": "#2ca02c", "SSP2": "#1f77b4", "SSP3": "#d62728", "SSP5": "#ff7f0e"}
    linestyles = {"SSP1": "-", "SSP2": "--", "SSP3": "-.", "SSP5": ":"}

    country_aliases = {
        "United States": ["United States", "United States of America"],
    }

    def select_country_rows(df: pd.DataFrame, country: str) -> pd.DataFrame:
        candidates = country_aliases.get(country, [country])
        for name in candidates:
            matched = df[df["name"] == name]
            if not matched.empty:
                return matched
        return df[df["name"] == country]

    def build_usa_estimated_series(df: pd.DataFrame, end_year: int = 2100, window: int = 10) -> pd.DataFrame:
        # 使用 OWID 历史人口进行指数外推，作为 USA 缺失的替代曲线
        usa = df[(df["country"] == "United States") & df["population"].notna()].sort_values("year")
        if usa.empty:
            return pd.DataFrame(columns=["year", "population"])

        hist = usa[["year", "population"]].copy()
        hist["population"] = hist["population"] / 1e6

        recent = usa.tail(window)
        if len(recent) < 2:
            return hist

        start = recent.iloc[0]
        end = recent.iloc[-1]
        years = int(end["year"] - start["year"])
        if years <= 0 or start["population"] <= 0:
            return hist

        growth_rate = (end["population"] / start["population"]) ** (1 / years) - 1
        last_year = int(end["year"])
        future_years = np.arange(last_year + 1, end_year + 1)
        if future_years.size == 0:
            return hist

        future_pop = end["population"] * (1 + growth_rate) ** (future_years - last_year)
        proj = pd.DataFrame({"year": future_years, "population": future_pop / 1e6})
        return pd.concat([hist, proj], ignore_index=True)

    all_ssp = pd.concat(
        [wb_data[f"ssp_{s}"] for s in ssp_order if f"ssp_{s}" in wb_data],
        ignore_index=True,
    )
    all_ssp["population"] = all_ssp["population"] / 1e6  # 转换为百万人
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, country in enumerate(target_countries):
        ax = axes[i]
        # 历史数据
        if "hist_pop" in wb_data:
            hist = wb_data["hist_pop"]
            hist_row = select_country_rows(hist, country)
            if not hist_row.empty and "population_2000" in hist_row.columns:
                ax.scatter(2000, hist_row["population_2000"].values[0] / 1e6,
                          color="black", s=50, zorder=5, label="历史观测（2000）")

        cdf = select_country_rows(all_ssp, country)
        has_ssp = (not cdf.empty) and cdf["population"].notna().any()
        if country == "United States" and not has_ssp and owid_df is not None:
            est = build_usa_estimated_series(owid_df)
            if not est.empty:
                ax.plot(est["year"], est["population"],
                        color="gray", linestyle="--", linewidth=1.8, label="USA（估计）")
        else:
            for ssp in ssp_order:
                sdf = cdf[cdf["ssp"] == ssp].sort_values("year")
                if not sdf.empty:
                    ax.plot(sdf["year"], sdf["population"],
                           color=colors.get(ssp, "gray"),
                           linestyle=linestyles.get(ssp, "-"),
                           linewidth=1.5, label=ssp)

        ax.set_title(f"{country}", fontsize=13)
        ax.set_xlabel("年份")
        ax.set_ylabel("人口（百万）")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("不同 SSP 情景下人口预测对比（2010—2100）", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig8_ssp_population.svg")
    plt.close(fig)
    print("图8 已保存")


# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("全球碳排放与能源结构时空可视化分析")
    print("=" * 60)

    # ---- 1. 加载数据 ----
    print("\n[1/7] 加载数据...")
    raw = load_owid_data(DATA_DIR / "owid-co2-data.csv")
    countries, regional, world = classify_locations(raw)
    print(f"  总行数: {len(raw)}, 国家实体: {len(countries)}, 区域汇总: {len(regional)}")

    # ---- 2. 数据清洗 ----
    print("\n[2/7] 数据清洗...")
    n_dupes = check_duplicates(raw)
    print(f"  重复记录: {n_dupes} 条")

    # 异常值检测
    outlier_cols = ["co2", "co2_per_capita", "energy_per_capita", "gdp"]
    outlier_counts = {}
    for col in outlier_cols:
        if col in countries.columns:
            valid = countries[col].notna()
            n_out = detect_outliers(countries[valid], col).sum()
            if n_out > 0:
                outlier_counts[col] = n_out

    # 质量报告
    print_quality_report(raw, countries, outlier_counts)

    # ---- 3. 特征构建 ----
    print("[3/7] 特征构建...")
    analysis_df = build_features(clean_country_data(countries))
    analysis_df = pd.concat([analysis_df, world], ignore_index=True)

    # 打印关键统计
    print("\n选定国家最新年份关键指标:")
    latest_yr = analysis_df["year"].max()
    display = analysis_df[
        (analysis_df["country"].isin(SELECTED_COUNTRIES))
        & (analysis_df["year"] == latest_yr)
    ][["country", "co2", "co2_per_capita", "co2_growth_prct",
       "gdp_growth", "decoupling_index"]]
    print(display.to_string(index=False))

    # ---- 4. 生成图表 ----
    print(f"\n[4/7] 生成 matplotlib 图表...")
    fig1_co2_trend(analysis_df)
    fig2_co2_per_capita(analysis_df)
    fig3_energy_source(analysis_df)
    fig6_ranking(analysis_df)
    fig7_energy_structure(analysis_df)

    # ---- 5. 生成 plotly 交互图表 ----
    print("\n[5/7] 生成 plotly 交互图表...")
    fig4_gdp_co2_scatter(analysis_df)
    fig5_world_map(analysis_df)

    # ---- 6. 补充数据处理 ----
    print("\n[6/7] 处理 World Bank 气候风险数据...")
    wb_path = DATA_DIR / "pop-x1_timeseries_pov550,popcount,popdensity_timeseries_annual_2000-2000,2010-2100_mean_SSP1,SSP2,SSP3,SSP4,SSP5,historical,ssp4,ssp119,ssp126,ssp245,ssp370,ssp585_gini,gpw-v4_poverty,rev11_mean.xlsx"
    if wb_path.exists():
        wb_data = load_worldbank_data(str(wb_path))
        fig8_ssp_population(wb_data, analysis_df)
        print(f"  加载了 {len(wb_data)} 个数据表")
    else:
        print(f"  World Bank 数据文件未找到: {wb_path}")

    # ---- 7. 完成 ----
    print("\n[7/7] 全部完成!")
    print(f"  图表保存至: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
