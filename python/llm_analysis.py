# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Prophet LLM 智能分析模块
========================
使用大语言模型（LLM）对 Prophet 预测结果进行自然语言智能解读。

支持的 LLM Provider:
- OpenAI (GPT-4o, GPT-4, GPT-3.5-turbo)
- Anthropic Claude (claude-3-5-sonnet, claude-3-opus)
- 本地 Ollama 模型 (llama3, mistral, ...)

用法示例:
    from llm_analysis import ProphetLLMAnalyzer
    from fbprophet import Prophet
    import pandas as pd

    df = pd.read_csv('data.csv')
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=365)
    fcst = m.predict(future)

    analyzer = ProphetLLMAnalyzer(model='gpt-4o')
    report = analyzer.analyze(m, fcst)
    print(report)
"""

from __future__ import absolute_import, division, print_function

import json
import logging
import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger('fbprophet.llm_analysis')


# ---------------------------------------------------------------------------
# LLM 客户端工厂
# ---------------------------------------------------------------------------

def _get_openai_client(api_key=None, base_url=None):
    """创建 OpenAI 兼容客户端（支持 OpenAI / Azure / 第三方代理）。"""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "请先安装 openai 包：pip install openai"
        ) from exc

    kwargs = {"api_key": api_key or os.environ.get("OPENAI_API_KEY")}
    if base_url:
        kwargs["base_url"] = base_url
    elif os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    return OpenAI(**kwargs)


def _get_anthropic_client(api_key=None):
    """创建 Anthropic Claude 客户端。"""
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "请先安装 anthropic 包：pip install anthropic"
        ) from exc

    return anthropic.Anthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
    )


def _get_ollama_client(base_url="http://localhost:11434"):
    """创建本地 Ollama 客户端（使用 OpenAI 兼容接口）。"""
    return _get_openai_client(api_key="ollama", base_url=f"{base_url}/v1")


# ---------------------------------------------------------------------------
# 统计摘要提取
# ---------------------------------------------------------------------------

class ProphetSummary:
    """从已拟合的 Prophet 模型和预测结果中提取关键统计摘要。"""

    def __init__(self, model, forecast):
        """
        Parameters
        ----------
        model : Prophet
            已调用 fit() 的 Prophet 模型实例。
        forecast : pd.DataFrame
            Prophet.predict() 返回的预测 DataFrame。
        """
        self.model = model
        self.forecast = forecast

    def history_summary(self):
        """历史数据统计摘要。"""
        hist = self.model.history
        if hist is None:
            return {}
        return {
            "data_start": str(hist['ds'].min().date()),
            "data_end": str(hist['ds'].max().date()),
            "n_points": int(len(hist)),
            "y_mean": float(round(hist['y'].mean(), 4)),
            "y_std": float(round(hist['y'].std(), 4)),
            "y_min": float(round(hist['y'].min(), 4)),
            "y_max": float(round(hist['y'].max(), 4)),
        }

    def trend_summary(self):
        """趋势相关摘要。"""
        fcst = self.forecast
        hist_end = self.model.history['ds'].max() if self.model.history is not None else None

        # 分历史段和未来段
        if hist_end is not None:
            hist_fcst = fcst[fcst['ds'] <= hist_end]
            future_fcst = fcst[fcst['ds'] > hist_end]
        else:
            hist_fcst = fcst
            future_fcst = pd.DataFrame()

        result = {
            "growth_type": self.model.growth,
            "n_changepoints_detected": int(len(self.model.changepoints_t))
            if self.model.changepoints_t is not None else 0,
        }

        # 变点日期（最多展示 5 个最显著的）
        if (self.model.params and 'delta' in self.model.params
                and self.model.changepoints_t is not None):
            deltas = np.array(self.model.params['delta']).flatten()
            cp_ts = self.model.changepoints_t
            if len(deltas) > 0 and len(cp_ts) > 0:
                # 转回日期
                start = self.model.start
                t_scale = self.model.t_scale
                cp_dates = pd.to_datetime(
                    start + cp_ts * t_scale
                )
                top_n = min(5, len(deltas))
                top_idx = np.argsort(np.abs(deltas))[-top_n:][::-1]
                result["top_changepoints"] = [
                    {
                        "date": str(cp_dates[i].date()),
                        "delta": float(round(deltas[i], 6)),
                    }
                    for i in top_idx
                ]

        # 预测趋势方向
        if not future_fcst.empty:
            trend_start = future_fcst['trend'].iloc[0]
            trend_end = future_fcst['trend'].iloc[-1]
            result["forecast_trend_start"] = float(round(trend_start, 4))
            result["forecast_trend_end"] = float(round(trend_end, 4))
            result["forecast_trend_direction"] = (
                "上升" if trend_end > trend_start else "下降"
            )

        return result

    def seasonality_summary(self):
        """季节性摘要。"""
        cols = self.forecast.columns.tolist()
        seasonality_info = {}
        for name in self.model.seasonalities:
            if name in cols:
                vals = self.forecast[name].dropna()
                seasonality_info[name] = {
                    "amplitude": float(round(vals.std() * 2, 4)),
                    "mode": self.model.seasonalities[name].get('mode', 'additive'),
                }
        return seasonality_info

    def forecast_summary(self):
        """预测结果摘要。"""
        fcst = self.forecast
        hist_end = self.model.history['ds'].max() if self.model.history is not None else None
        if hist_end is not None:
            future_fcst = fcst[fcst['ds'] > hist_end].copy()
        else:
            future_fcst = fcst.copy()

        if future_fcst.empty:
            return {}

        return {
            "forecast_start": str(future_fcst['ds'].min().date()),
            "forecast_end": str(future_fcst['ds'].max().date()),
            "forecast_periods": int(len(future_fcst)),
            "yhat_mean": float(round(future_fcst['yhat'].mean(), 4)),
            "yhat_min": float(round(future_fcst['yhat'].min(), 4)),
            "yhat_max": float(round(future_fcst['yhat'].max(), 4)),
            "uncertainty_width_mean": float(round(
                (future_fcst['yhat_upper'] - future_fcst['yhat_lower']).mean(), 4
            )) if 'yhat_upper' in future_fcst.columns else None,
        }

    def to_dict(self):
        """汇总所有摘要为字典。"""
        return {
            "history": self.history_summary(),
            "trend": self.trend_summary(),
            "seasonalities": self.seasonality_summary(),
            "forecast": self.forecast_summary(),
            "model_params": {
                "changepoint_prior_scale": self.model.changepoint_prior_scale,
                "seasonality_prior_scale": self.model.seasonality_prior_scale,
                "seasonality_mode": self.model.seasonality_mode,
                "interval_width": self.model.interval_width,
                "mcmc_samples": self.model.mcmc_samples,
            },
        }

    def to_json(self, indent=2):
        """输出为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# LLM Prompt 模板
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    你是一位专业的时间序列数据分析师，精通 Facebook Prophet 预测模型。
    你的任务是根据 Prophet 模型的统计摘要，生成清晰、准确、有洞察力的中文分析报告。
    报告应面向业务决策者，避免过多数学术语，重点说明业务含义和行动建议。
""")

ANALYSIS_PROMPT_TEMPLATE = textwrap.dedent("""\
    以下是 Prophet 时间序列预测模型的关键统计摘要（JSON 格式）：

    ```json
    {summary_json}
    ```

    请基于上述数据，生成一份结构化的中文智能分析报告，包含以下章节：

    ## 1. 数据概况
    - 历史数据时间范围、数据量、数值分布特征

    ## 2. 趋势分析
    - 增长类型（线性/逻辑增长）
    - 重要变点时间节点及可能的业务原因
    - 未来预测期趋势方向与幅度

    ## 3. 季节性分析
    - 各周期（年度/周度/日度）季节性强度
    - 规律性描述及业务解读

    ## 4. 预测结果解读
    - 预测区间范围与置信度
    - 预测期内峰值/谷值时间及数值
    - 预测不确定性评估

    ## 5. 风险提示
    - 模型局限性说明
    - 需要关注的潜在异常或风险点

    ## 6. 行动建议
    - 基于预测结果的 2-3 条具体业务建议

    请用简洁专业的语言撰写，每章不超过 150 字。
""")


# ---------------------------------------------------------------------------
# 主分析器类
# ---------------------------------------------------------------------------

class ProphetLLMAnalyzer:
    """
    使用 LLM 对 Prophet 预测结果进行智能分析的主类。

    Parameters
    ----------
    model : str
        LLM 模型名称，例如：
        - OpenAI: "gpt-4o", "gpt-4", "gpt-3.5-turbo"
        - Anthropic: "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"
        - Ollama: "llama3", "mistral", "qwen2"
    provider : str, optional
        LLM 提供商："openai"（默认）, "anthropic", "ollama"。
        若未指定，则根据 model 名称自动推断。
    api_key : str, optional
        API 密钥。未指定时读取环境变量 OPENAI_API_KEY 或 ANTHROPIC_API_KEY。
    base_url : str, optional
        自定义 API Base URL（用于代理或本地部署）。
    max_tokens : int, optional
        LLM 最大输出 Token 数，默认 2000。
    temperature : float, optional
        LLM 生成温度，默认 0.3（偏保守、准确）。

    Examples
    --------
    >>> analyzer = ProphetLLMAnalyzer(model='gpt-4o')
    >>> report = analyzer.analyze(m, fcst)
    >>> print(report)

    >>> # 使用本地 Ollama
    >>> analyzer = ProphetLLMAnalyzer(model='llama3', provider='ollama')
    >>> report = analyzer.analyze(m, fcst)
    """

    # 根据模型名前缀自动推断 provider
    _PROVIDER_HINTS = {
        "gpt": "openai",
        "o1": "openai",
        "o3": "openai",
        "claude": "anthropic",
        "llama": "ollama",
        "mistral": "ollama",
        "qwen": "ollama",
        "gemma": "ollama",
        "phi": "ollama",
    }

    def __init__(
        self,
        model="gpt-4o",
        provider=None,
        api_key=None,
        base_url=None,
        max_tokens=2000,
        temperature=0.3,
    ):
        self.model = model
        self.provider = provider or self._infer_provider(model)
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _infer_provider(self, model):
        """根据模型名称前缀推断 provider。"""
        model_lower = model.lower()
        for prefix, prov in self._PROVIDER_HINTS.items():
            if model_lower.startswith(prefix):
                return prov
        return "openai"

    def _build_client(self):
        """懒加载 LLM 客户端。"""
        if self._client is not None:
            return self._client

        if self.provider == "anthropic":
            self._client = _get_anthropic_client(api_key=self.api_key)
        elif self.provider == "ollama":
            self._client = _get_ollama_client(
                base_url=self.base_url or "http://localhost:11434"
            )
        else:
            self._client = _get_openai_client(
                api_key=self.api_key, base_url=self.base_url
            )
        return self._client

    def _call_llm(self, user_prompt):
        """调用 LLM 并返回文本结果。"""
        client = self._build_client()

        if self.provider == "anthropic":
            import anthropic
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return message.content[0].text

        # OpenAI 兼容接口（openai / ollama）
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def extract_summary(self, model, forecast):
        """
        从 Prophet 模型和预测结果中提取统计摘要。

        Parameters
        ----------
        model : Prophet
            已拟合的 Prophet 模型。
        forecast : pd.DataFrame
            Prophet.predict() 输出的预测 DataFrame。

        Returns
        -------
        dict
            统计摘要字典。
        """
        return ProphetSummary(model, forecast).to_dict()

    def analyze(self, model, forecast, extra_context=None):
        """
        对 Prophet 预测结果进行 LLM 智能分析，返回中文报告文本。

        Parameters
        ----------
        model : Prophet
            已拟合的 Prophet 模型实例。
        forecast : pd.DataFrame
            Prophet.predict() 返回的预测 DataFrame。
        extra_context : str, optional
            额外的业务背景信息（会追加到 prompt 中）。

        Returns
        -------
        str
            LLM 生成的中文分析报告。
        """
        summary = ProphetSummary(model, forecast)
        summary_json = summary.to_json()

        user_prompt = ANALYSIS_PROMPT_TEMPLATE.format(summary_json=summary_json)

        if extra_context:
            user_prompt += f"\n\n**额外业务背景**：\n{extra_context}"

        logger.info(
            "调用 LLM (%s / %s) 进行 Prophet 预测智能分析...",
            self.provider,
            self.model,
        )
        report = self._call_llm(user_prompt)
        logger.info("LLM 分析完成，报告长度: %d 字", len(report))
        return report

    def analyze_anomalies(self, model, forecast, threshold=2.0):
        """
        识别历史数据中的异常点并用 LLM 给出解释。

        Parameters
        ----------
        model : Prophet
            已拟合的 Prophet 模型。
        forecast : pd.DataFrame
            Prophet.predict() 返回的预测 DataFrame（包含历史段）。
        threshold : float
            异常判断阈值（残差标准差倍数），默认 2.0。

        Returns
        -------
        str
            LLM 生成的异常分析报告。
        """
        hist = model.history
        if hist is None:
            raise ValueError("模型尚未拟合，请先调用 fit()。")

        # 合并历史 y 与预测 yhat
        hist_fcst = forecast[forecast['ds'].isin(hist['ds'])].copy()
        hist_fcst = hist_fcst.merge(
            hist[['ds', 'y']], on='ds', how='left'
        )
        hist_fcst['residual'] = hist_fcst['y'] - hist_fcst['yhat']
        residual_std = hist_fcst['residual'].std()
        anomalies = hist_fcst[
            hist_fcst['residual'].abs() > threshold * residual_std
        ][['ds', 'y', 'yhat', 'residual']].copy()
        anomalies['ds'] = anomalies['ds'].dt.strftime('%Y-%m-%d')
        anomalies = anomalies.round(4)

        anomaly_info = {
            "threshold_sigma": threshold,
            "residual_std": float(round(residual_std, 4)),
            "n_anomalies": int(len(anomalies)),
            "anomaly_points": anomalies.to_dict(orient='records'),
        }

        prompt = textwrap.dedent(f"""\
            Prophet 模型在历史数据中检测到如下异常点（残差超过 {threshold}σ）：

            ```json
            {json.dumps(anomaly_info, ensure_ascii=False, indent=2)}
            ```

            请分析：
            1. 这些异常点的时间分布规律（集中在某些月份/季节？）
            2. 可能的业务原因或外部事件
            3. 这些异常是否会影响预测准确性
            4. 建议的处理方式（保留/移除/使用 holidays 参数建模）

            用中文简洁回答，总长度不超过 300 字。
        """)

        return self._call_llm(prompt)

    def compare_scenarios(self, scenarios):
        """
        对比多个 Prophet 预测场景并给出综合建议。

        Parameters
        ----------
        scenarios : dict
            场景字典，格式为 {场景名: {"model": Prophet实例, "forecast": DataFrame}}。
            例如：{"乐观": {"model": m1, "forecast": f1},
                   "保守": {"model": m2, "forecast": f2}}

        Returns
        -------
        str
            LLM 生成的多场景对比分析报告。
        """
        scenario_summaries = {}
        for name, data in scenarios.items():
            summary = ProphetSummary(data["model"], data["forecast"])
            scenario_summaries[name] = summary.forecast_summary()

        prompt = textwrap.dedent(f"""\
            以下是多个 Prophet 预测场景的对比摘要：

            ```json
            {json.dumps(scenario_summaries, ensure_ascii=False, indent=2)}
            ```

            请生成多场景对比分析报告，包含：
            1. 各场景预测区间对比（表格形式展示关键指标）
            2. 场景间差异的核心驱动因素
            3. 综合建议：在什么条件下各场景更可能发生
            4. 推荐的决策参考场景及理由

            用中文撰写，结构清晰，总长度不超过 400 字。
        """)

        return self._call_llm(prompt)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def analyze_prophet(model, forecast, llm_model="gpt-4o", **kwargs):
    """
    便捷函数：一行完成 Prophet 预测结果的 LLM 智能分析。

    Parameters
    ----------
    model : Prophet
        已拟合的 Prophet 模型。
    forecast : pd.DataFrame
        Prophet.predict() 返回的预测 DataFrame。
    llm_model : str
        LLM 模型名称，默认 "gpt-4o"。
    **kwargs
        传递给 ProphetLLMAnalyzer 的其他参数。

    Returns
    -------
    str
        分析报告文本。

    Examples
    --------
    >>> from llm_analysis import analyze_prophet
    >>> report = analyze_prophet(m, fcst, llm_model="gpt-4o")
    >>> print(report)
    """
    analyzer = ProphetLLMAnalyzer(model=llm_model, **kwargs)
    return analyzer.analyze(model, forecast)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def _cli():
    """命令行工具：从 CSV 文件读取 Prophet 预测结果并调用 LLM 分析。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prophet LLM 智能分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              python llm_analysis.py --data data.csv --periods 365 --model gpt-4o
              python llm_analysis.py --data data.csv --periods 90 --model llama3 --provider ollama
        """),
    )
    parser.add_argument("--data", required=True, help="输入 CSV 文件路径（需含 ds 和 y 列）")
    parser.add_argument("--periods", type=int, default=365, help="预测未来天数（默认 365）")
    parser.add_argument("--freq", default="D", help="时间序列频率（默认 D=日）")
    parser.add_argument("--model", default="gpt-4o", help="LLM 模型名称（默认 gpt-4o）")
    parser.add_argument("--provider", default=None, help="LLM Provider (openai/anthropic/ollama)")
    parser.add_argument("--api-key", default=None, help="API 密钥（默认读环境变量）")
    parser.add_argument("--base-url", default=None, help="自定义 API Base URL")
    parser.add_argument("--output", default=None, help="输出报告文件路径（默认打印到终端）")
    parser.add_argument("--context", default=None, help="额外业务背景信息")
    args = parser.parse_args()

    # 导入 Prophet（延迟导入）
    try:
        from fbprophet import Prophet
    except ImportError:
        from prophet import Prophet  # 新版包名

    logger.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("读取数据: %s", args.data)
    df = pd.read_csv(args.data)
    df['ds'] = pd.to_datetime(df['ds'])

    logger.info("训练 Prophet 模型...")
    m = Prophet()
    m.fit(df)

    logger.info("生成未来 %d 期预测...", args.periods)
    future = m.make_future_dataframe(periods=args.periods, freq=args.freq)
    fcst = m.predict(future)

    logger.info("调用 LLM 进行智能分析...")
    analyzer = ProphetLLMAnalyzer(
        model=args.model,
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    report = analyzer.analyze(m, fcst, extra_context=args.context)

    # 输出报告
    header = (
        f"# Prophet 预测智能分析报告\n"
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"LLM 模型: {args.model}\n\n"
        f"---\n\n"
    )
    full_report = header + report

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(full_report)
        logger.info("报告已保存至: %s", args.output)
    else:
        print(full_report)


if __name__ == "__main__":
    _cli()
