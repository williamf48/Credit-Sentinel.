# CreditSentinel

**A Streamlit prototype for interactive credit-risk scoring and financial-ratio analysis.**

The original CreditSentinel application uses a saved XGBoost classifier, financial-ratio inputs and sector-based heuristic adjustments to generate a dashboard score. It supports manual analysis and issuer selection from a local cache, with financial charts where historical records are available.

Built by [William Fernandes](https://github.com/williamf48).

**Project relationship:** this repository contains the earlier single-model application. [CreditSentinel V2](https://github.com/williamf48/CreditSentinelV2) extends the approach with XGBoost, Random Forest and Logistic Regression.

## Features

- XGBoost model inference on four financial inputs.
- Manual entry and selection from cached issuer records.
- Sector-dependent leverage thresholds.
- Transparent leverage and promoter-pledge score adjustments.
- Score gauge and rule-based credit memo.
- Historical leverage and annual revenue charts when available.
- Cached market capitalization and 52-week price information.

## Technology stack

| Purpose | Technologies |
| --- | --- |
| Interface | Streamlit |
| Model inference | XGBoost |
| Data processing | Pandas, NumPy |
| Charts | Plotly |
| Local data | JSON cache |

The dependency file also includes yfinance and scikit-learn. Issuer data is read from a local cache; the current app does not fetch live market data.

## Scoring method

| Financial input | Model feature |
| --- | --- |
| Debt / EBITDA | `Debt_to_EBITDA` |
| Interest coverage | `Interest_Coverage` |
| Promoter pledge (%) | `Pledged_Pct` |
| Current ratio | `Current_Ratio` |

The application obtains the positive-class probability from the saved classifier, then adds `0.20` for leverage above the sector threshold and `0.10` for promoter pledge above 20%. It clips the result to `[0.01, 0.99]` and displays it on a 0–100 scale.

| Display score | Current UI label |
| --- | --- |
| Below 35 | INVESTMENT GRADE (IG) |
| 35 to below 70 | SPECULATIVE / HY |
| 70 and above | DISTRESSED / DEFAULT |

The adjusted score is a prototype risk indicator. Its calibration as a default probability is not established by this repository. The labels are internal display bands, not external credit ratings or verified default events. Sector thresholds are heuristic defaults rather than borrower-specific contractual covenants.

## Repository contents

| File | Purpose |
| --- | --- |
| `app.py` | Dashboard, inputs, scoring and charts |
| `creditsentinel_model.json` | Saved XGBoost classifier |
| `stock_cache.json` | Cached issuer financial inputs and optional histories |
| `requirements.txt` | Python dependencies |
| `.devcontainer/devcontainer.json` | Development-container configuration |

## Run locally

### 1. Clone and create an environment

```bash
git clone https://github.com/williamf48/Credit-Sentinel..git
cd Credit-Sentinel.
python -m venv .venv
```

The clone URL contains two consecutive periods before `git` because the repository name ends in a period. On Windows, use a directory name without the trailing period:

```powershell
git clone https://github.com/williamf48/Credit-Sentinel..git CreditSentinelOriginal
cd CreditSentinelOriginal
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2. Install and launch

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the URL printed by Streamlit, usually http://localhost:8501.

Run from the cloned repository directory because model and cache paths are relative. The model file must be present for startup; without the cache, manual entry remains available.

## Using the application

1. Select an issuer or **Manual Entry**.
2. Review or enter the financial ratios.
3. Click **Run Credit Model**.
4. Inspect the score, credit memo and available charts.

Issuer selection can also trigger scoring automatically. Chart availability depends on the cached issuer's history and revenue fields.

## Scope and limitations

- This repository contains inference code and a saved model, not the training workflow.
- Dataset provenance, preprocessing, model selection and held-out evaluation are not documented here.
- No performance metrics are claimed without reproducible evaluation evidence.
- Cache freshness is not guaranteed and no cache-refresh script is included.
- Dependencies are unpinned and no automated test suite is included.
- The heuristic adjustments and display thresholds require separate validation for any real credit-assessment use.

## Related projects

- [CreditSentinel V2](https://github.com/williamf48/CreditSentinelV2): three-model ensemble successor.
- [RiskVision](https://github.com/williamf48/RiskVision): borrower deterioration, covenant and portfolio monitoring.

## License

No license file is currently included in this repository.

