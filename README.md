全球碳排放与能源结构时空可视化分析
====================================

本项目基于 OWID CO2 数据集（1990–2024）与世界银行 SSP 人口预测数据，对全球及 12 个代表性国家的碳排放演进特征进行时空可视化分析，输出 8 张核心图表与论文排版文件。

项目结构
--------

- main.py: 数据处理与图表生成主程序
- paper.typ: 论文正文（Typst）
- lib.typ: Typst 模板与样式配置
- data/: 数据文件（OWID CSV + World Bank XLSX）
- output/: 图表与中间产物输出目录

图表输出
--------

Matplotlib 生成的图表已统一输出为 SVG，Plotly 图表输出为 HTML：

- output/fig1_co2_trend.svg
- output/fig2_co2_per_capita.svg
- output/fig3_energy_source.svg
- output/fig6_ranking.svg
- output/fig7_energy_structure.svg
- output/fig8_ssp_population.svg
- output/fig4_gdp_co2_scatter.html
- output/fig5_world_map.html

环境与依赖
----------

- Python 3.12+
- 主要依赖: pandas, numpy, matplotlib, seaborn, plotly, openpyxl
- 推荐使用 uv 管理环境

快速开始
--------

1) 安装依赖

```bash
uv sync
```

2) 运行主程序生成图表

```bash
uv run python main.py
```

3) 生成的图表位于 output/ 目录

4) 编译PDF

```bash
typst compile paper.typ 
```

数据说明
--------

- OWID CO2 主数据: data/owid-co2-data.csv
- OWID 字段说明: data/owid-co2-codebook.csv
- World Bank SSP 人口预测: data/pop-x1_timeseries_*.xlsx

注意事项
--------

- 若系统缺少中文字体，Matplotlib 可能出现中文乱码。建议安装 SimHei 或 Noto Sans SC。
- 图8中的美国人口序列来自 OWID 历史人口的指数外推，图例标注为“USA（估计）”。

常见问题
--------

1) 运行报错找不到依赖
	- 请确认已执行 uv sync 并使用 uv run 运行。

2) 图表未生成
	- 请确认 data/ 下数据文件完整且路径未变更。
