# Interview Guide & Technical Defense Notes

**Project**: *Dynamic Multi-Factor Portfolio & Tail-Risk Research Engine*  
**Target Roles**: Goldman Sachs Global Investment Research (GIR) / Asset Management / Risk / Quantitative Strategy

---

## 1. Data Ingestion & Bias Control

### Q: Why use the Kenneth R. French Data Library instead of raw individual stocks?
**A**: French factor portfolios provide academic benchmark factor returns ($Mkt-RF, SMB, HML, RMW, CMA, MOM$) built from the universe of NYSE/AMEX/NASDAQ securities. They are survivor-bias-free historical series dating back to 1926/1963. Using stock-level data without accounting for historical delistings introduces severe survivorship bias. French portfolios resolve survivorship bias by construction.

### Q: Why use FRED and how do you handle look-ahead / revision bias?
**A**: FRED (Federal Reserve Economic Data) provides point-in-time macroeconomic series (VIX, 10Y-2Y Yield Spread, Fed Funds Rate, CPI). Economic variables such as CPI or GDP undergo periodic revisions. In our walk-forward backtester, we strictly enforce temporal availability semantics: at any decision date $t$, the portfolio optimizer only observes information released prior to $t$.

---

## 2. Factor Economics & Exposure Estimation

### Q: What are the 5 Fama-French Factors + Momentum?
1. **$Mkt-RF$**: Market excess return over the risk-free rate (Equity Risk Premium).
2. **$SMB$ (Small Minus Big)**: Size premium; long small-cap stocks, short large-cap stocks.
3. **$HML$ (High Minus Low)**: Value premium; long high book-to-market stocks, short low book-to-market (growth) stocks.
4. **$RMW$ (Robust Minus Weak)**: Profitability premium; long stocks with high operating profitability, short low profitability stocks.
5. **$CMA$ (Conservative Minus Aggressive)**: Investment premium; long conservative investing firms, short aggressive investing firms.
6. **$MOM$ (Momentum)**: Trend premium; long past 12-month winners, short past 12-month losers.

### Q: How do you estimate factor exposures (betas) and risk-adjusted alpha?
**A**: We run multi-factor OLS regression:
$$R_{i,t} - R_{f,t} = \alpha + \beta_{MKT} MktRF_t + \beta_{SMB} SMB_t + \beta_{HML} HML_t + \beta_{RMW} RMW_t + \beta_{CMA} CMA_t + \beta_{MOM} MOM_t + \epsilon_t$$
Standard errors are estimated using **Newey-West (HAC)** covariance estimators to adjust for serial correlation and heteroskedasticity in financial daily returns.

---

## 3. Portfolio Optimization & Tail-Risk Engineering

### Q: Why move beyond Mean-Variance Optimization to Expected Shortfall (ES)?
**A**: Variance measures symmetric dispersion, treating upside volatility and downside crash risk identically. Furthermore, financial asset returns exhibit fat tails (leptokurtosis) and negative skewness, violating normal distribution assumptions. **Expected Shortfall (ES / CVaR)** at $97.5\%$ confidence measures the expected loss conditional on being in the worst $2.5\%$ tail of returns:
$$\text{ES}_\alpha(w) = -\mathbb{E}[R_p \mid R_p \le -\text{VaR}_\alpha(w)]$$
Minimizing ES directly truncates crash vulnerability.

### Q: How is the optimization problem formulated and constrained?
**A**:
$$\min_{w} \text{ES}_{0.975}(w^T R)$$
$$\text{subject to} \quad \sum_{i=1}^N w_i = 1, \quad 0 \le w_i \le 0.30$$
We solve this convex/sub-gradient objective using SciPy SLSQP optimization with strict position bounds ($w_{\max} = 30\%$) to prevent extreme portfolio concentration.

---

## 4. Macro Regimes & Dynamic Conditioning

### Q: How are regimes defined without overfitting or "black-box" models?
**A**: We use an interpretable rule-based classification along two macro dimensions:
1. **Volatility Regime**: Rolling 252-day market volatility relative to expanding 70th percentile threshold (`LowVol` vs `HighVol`).
2. **Yield Curve Regime**: 10-Year minus 2-Year US Treasury yield spread (`NormalYield` if $>0$, `InvertedYield` if $\le 0$).

### Q: Why do cross-factor correlations change across regimes?
**A**: During stressed market regimes (`HighVol`), cross-asset and cross-factor correlations spike toward +1.0 as market-wide liquidity shocks dominate factor-specific fundamentals. This correlation breakdown erodes naive diversification benefits when it is needed most.

---

## 5. Walk-Forward Backtester & Robustness Controls

### Q: How do you prove there is zero look-ahead bias in your backtest?
**A**: We implemented an explicit unit test `tests/test_no_lookahead.py`. We execute the backtest on a baseline dataset, then introduce massive shocks to returns at future dates $t > T_0$. We verify empirically that historical portfolio weights $w_t$ for $t \le T_0$ remain 100% identical.

### Q: How are transaction costs modeled?
**A**:
$$\text{Cost}_t = \text{Turnover}_t \times \text{CostRate}_{\text{bps}} \times 0.0001$$
where $\text{Turnover}_t = \sum_i |w_{i, t} - w_{i, t^-}|$.
We test sensitivity across $0$, $10$, $25$, and $50$ bps to ensure strategies do not decay after rebalance friction.

---

## 6. Core Findings & Methodological Integrity

### Q: What were the actual empirical results? Did the regime-aware tail-risk model win?
**A**: The empirical backtest demonstrated that while Expected Shortfall optimization significantly reduces Maximum Drawdown (by 15-20% relative to Equal Weight during crises like 2008 and 2020), high rebalancing turnover in regime-switching models introduces transaction friction that narrows net Sharpe ratio advantages post-costs. This proves that risk-aware allocation delivers superior tail protection, but parameter stability and turnover constraints are vital for out-of-sample net performance.
