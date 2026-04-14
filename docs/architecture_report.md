# Prophet 核心架构文档报告

> **项目**：Facebook Prophet — 自动时间序列预测框架  
> **版本**：0.5  
> **语言**：Python / R  
> **文档日期**：2026-04-14

---

## 一、项目概述

Prophet 是由 Facebook Core Data Science 团队开源的时间序列预测工具，基于**加法模型**（Additive Model）对时间序列进行分解与预测。它适用于具有强季节性规律和多年历史数据的业务场景，对缺失值和异常值有较好的鲁棒性。

### 核心特性

| 特性 | 说明 |
|------|------|
| 趋势模型 | 线性增长（linear）或逻辑增长（logistic） |
| 季节性 | 年度 / 周度 / 日度，支持自定义傅里叶阶数 |
| 节假日效应 | 内置 100+ 国家节日，支持自定义 |
| 变点检测 | 自动或手动指定趋势变点（changepoints） |
| 贝叶斯推断 | 底层使用 Stan，支持 MAP 或 MCMC 采样 |
| 不确定性估计 | 预测区间基于后验采样 |
| 额外回归量 | 支持添加外部特征变量 |

---

## 二、整体系统架构图

```mermaid
graph TB
    subgraph 数据输入层["📥 数据输入层"]
        INPUT["输入 DataFrame\nds列（日期）+ y列（数值）\n+ 可选: cap / floor / 额外回归量"]
    end

    subgraph 预处理层["🔧 预处理层 (setup_dataframe)"]
        VALID["数据校验\n• 日期格式检查\n• 无穷值/NaN检测\n• 时区检查"]
        SCALE["数据归一化\n• y_scale\n• t_scale\n• regressor 标准化"]
    end

    subgraph 模型构建层["🏗️ 模型构建层 (fit)"]
        subgraph 趋势模型["趋势模型 (Trend)"]
            LINEAR["线性趋势\n+ 变点机制"]
            LOGISTIC["逻辑趋势\n+ 增长上限 cap"]
        end

        subgraph 季节性模型["季节性模型 (Seasonality)"]
            YEARLY["年度季节性\nFourier(10阶)"]
            WEEKLY["周度季节性\nFourier(3阶)"]
            DAILY["日度季节性\nFourier(4阶)"]
            CUSTOM["自定义季节性"]
        end

        subgraph 节假日模型["节假日模型 (Holidays)"]
            BUILTIN["内置国家节日\nhdays.py"]
            CUSTOM_H["自定义节假日\nDataFrame"]
        end

        REGRESSORS["额外回归量\n(Extra Regressors)"]
    end

    subgraph 推断引擎["⚙️ 推断引擎 (Stan)"]
        STAN["Stan 贝叶斯模型\nprophet_model.pkl"]
        MAP["MAP 估计\n(mcmc_samples=0)"]
        MCMC["MCMC 采样\n(mcmc_samples>0)"]
    end

    subgraph 预测层["📈 预测层 (predict)"]
        FUTURE["生成未来日期\nmake_future_dataframe()"]
        PREDICT["预测计算\nyhat = trend + seasonality + holidays"]
        INTERVAL["不确定性区间\nyhat_lower / yhat_upper"]
    end

    subgraph 诊断层["🔍 诊断层 (diagnostics)"]
        CV["时序交叉验证\ncross_validation()"]
        METRICS["性能指标\nMAE / MAPE / RMSE / Coverage"]
    end

    subgraph 可视化层["📊 可视化层 (plot)"]
        MPLPLOT["Matplotlib 静态图\nplot() / plot_components()"]
        PLOTLY["Plotly 交互图\nplot_plotly()"]
    end

    INPUT --> VALID
    VALID --> SCALE
    SCALE --> 趋势模型
    SCALE --> 季节性模型
    SCALE --> 节假日模型
    SCALE --> REGRESSORS
    趋势模型 --> STAN
    季节性模型 --> STAN
    节假日模型 --> STAN
    REGRESSORS --> STAN
    STAN --> MAP
    STAN --> MCMC
    MAP --> PREDICT
    MCMC --> PREDICT
    FUTURE --> PREDICT
    PREDICT --> INTERVAL
    PREDICT --> CV
    CV --> METRICS
    PREDICT --> MPLPLOT
    PREDICT --> PLOTLY
```

---

## 三、模块文件结构

```
prophet/
├── python/
│   ├── fbprophet/
│   │   ├── forecaster.py      # 核心类 Prophet，fit/predict 主逻辑
│   │   ├── diagnostics.py     # 交叉验证、性能指标计算
│   │   ├── plot.py            # Matplotlib + Plotly 绘图函数
│   │   ├── make_holidays.py   # 节假日数据生成工具
│   │   ├── hdays.py           # 内置各国节假日定义
│   │   ├── models.py          # 加载预编译 Stan 模型
│   │   └── tests/
│   │       ├── test_prophet.py      # 核心功能测试
│   │       └── test_diagnostics.py  # 诊断模块测试
│   ├── requirements.txt
│   └── setup.py
├── R/                         # R 语言实现（与 Python 功能对等）
├── docs/                      # 文档（Jekyll 静态站点）
├── notebooks/                 # Jupyter 示例笔记本
├── examples/                  # 示例数据与脚本
└── stan/                      # Stan 贝叶斯模型源码
```

---

## 四、加法模型数学公式

Prophet 将时间序列分解为：

```
y(t) = trend(t) + seasonality(t) + holidays(t) + ε(t)
```

### 4.1 趋势模型

**线性增长（分段线性）**：
```
trend(t) = (k + a(t)ᵀδ) · t + (m + a(t)ᵀγ)
```
- `k`：基础增长率
- `δ`：变点处的增长率调整量
- `m`：基础偏移量

**逻辑增长（Logistic）**：
```
trend(t) = L / (1 + exp(-k(t - m)))
```
- `L`：增长上限（cap）

### 4.2 季节性模型（傅里叶级数）

```
seasonality(t) = Σ [aₙcos(2πnt/P) + bₙsin(2πnt/P)]
```
- `P`：周期（年=365.25，周=7）
- `N`：傅里叶阶数

### 4.3 变点检测

```mermaid
graph LR
    HIST["历史数据前 80%"] --> CP["自动选取 25 个候选变点"]
    CP --> SPARSE["稀疏先验 δ ~ Laplace(0, τ)"]
    SPARSE --> FIT["Stan 推断有效变点"]
```

---

## 五、数据流详解

```mermaid
sequenceDiagram
    participant User as 用户
    participant Prophet as Prophet 对象
    participant Stan as Stan 模型
    participant Predict as 预测引擎

    User->>Prophet: m = Prophet(**params)
    User->>Prophet: m.fit(df)
    Prophet->>Prophet: setup_dataframe(df, initialize_scales=True)
    Prophet->>Prophet: set_changepoints()
    Prophet->>Prophet: make_seasonality_features()
    Prophet->>Prophet: make_holiday_features()
    Prophet->>Stan: MCMC / MAP 推断
    Stan-->>Prophet: params (k, m, δ, β, σ)
    User->>Prophet: future = m.make_future_dataframe(periods=365)
    User->>Prophet: fcst = m.predict(future)
    Prophet->>Predict: setup_dataframe(future)
    Predict->>Predict: predict_trend()
    Predict->>Predict: predict_seasonal_components()
    Predict-->>User: fcst DataFrame (yhat, yhat_lower, yhat_upper, ...)
```

---

## 六、交叉验证架构

```mermaid
graph LR
    DATA["全量历史数据"] --> CUTOFFS["生成多个截止点\ngenerate_cutoffs()"]
    CUTOFFS --> FOLD1["训练集 1\n→ 预测未来 horizon"]
    CUTOFFS --> FOLD2["训练集 2\n→ 预测未来 horizon"]
    CUTOFFS --> FOLDK["训练集 K\n→ 预测未来 horizon"]
    FOLD1 --> METRICS
    FOLD2 --> METRICS
    FOLDK --> METRICS["汇总指标\nMAE / MAPE / RMSE / Coverage\nperformance_metrics()"]
```

---

## 七、类与函数速查

### Prophet 主类关键方法

| 方法 | 功能 |
|------|------|
| `fit(df)` | 拟合模型（调用 Stan 推断） |
| `predict(future)` | 生成预测结果 |
| `make_future_dataframe(periods, freq)` | 生成未来时间索引 |
| `add_seasonality(name, period, fourier_order)` | 添加自定义季节性 |
| `add_regressor(name)` | 添加额外回归变量 |
| `add_country_holidays(country_name)` | 添加国家节假日 |
| `plot(fcst)` | 绘制预测图 |
| `plot_components(fcst)` | 绘制分量图 |

### diagnostics 模块

| 函数 | 功能 |
|------|------|
| `cross_validation(model, horizon, period, initial)` | 时序交叉验证 |
| `performance_metrics(df, metrics, rolling_window)` | 计算评估指标 |
| `plot_cross_validation_metric(df_cv, metric)` | 可视化交叉验证指标 |

---

## 八、LLM 智能分析集成

本仓库提供了 `python/llm_analysis.py`，利用 LLM（大语言模型）对 Prophet 预测结果进行智能解读：

- **自动趋势解读**：识别趋势方向、变点时间节点及业务含义
- **季节性解读**：分析周度/年度规律并给出业务解释
- **异常检测解读**：对节假日效应和残差异常做自然语言说明
- **预测置信度评估**：对不确定性区间给出风险评估建议

> 详见 [`python/llm_analysis.py`](../python/llm_analysis.py)

---

## 九、快速开始

```python
from fbprophet import Prophet
import pandas as pd

# 1. 准备数据（ds + y 两列）
df = pd.read_csv('data.csv')
df['ds'] = pd.to_datetime(df['ds'])

# 2. 创建并训练模型
m = Prophet(
    changepoint_prior_scale=0.05,
    yearly_seasonality=True,
    weekly_seasonality=True,
)
m.add_country_holidays(country_name='CN')
m.fit(df)

# 3. 预测未来 365 天
future = m.make_future_dataframe(periods=365)
fcst = m.predict(future)

# 4. 可视化
fig = m.plot(fcst)
fig2 = m.plot_components(fcst)

# 5. LLM 智能分析（需配置 API Key）
from llm_analysis import ProphetLLMAnalyzer
analyzer = ProphetLLMAnalyzer(model="gpt-4o")
report = analyzer.analyze(m, fcst)
print(report)
```

---

*本文档由 `docs/architecture_report.md` 维护，架构图使用 Mermaid 语法，在 GitHub 上可直接渲染预览。*
