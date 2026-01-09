# FYERS ML Trading System

## Overview
This repository contains a **probability‑driven, risk‑adjusted algorithmic trading system** built using **machine learning** and the **FYERS API**.  
The system is designed as an **end‑to‑end trading pipeline**, covering data acquisition, feature engineering, walk‑forward model training, signal generation, risk management, and realistic backtesting.

The strategy prioritizes **Sharpe ratio and signal stability** over raw profitability, making it suitable for **low‑data regimes** and closer to real‑world trading conditions.

This project was developed as part of **Finstreet Problem Statement – Round 2**.

---

## Key Highlights
- 📈 **Probability‑based ML approach** (Logistic Regression)
- 🔄 **Walk‑forward (expanding window) training** to prevent data leakage
- ⚖️ **Sharpe‑optimized strategy design**
- 🛡️ **ATR‑based dynamic risk management**
- 🧪 **Realistic backtesting with capital tracking**
- 🔌 **FYERS API integration (data + execution logic)**

---

## Strategy Summary
- The model predicts the **probability of an upward price movement** for the next trading day.
- Trades are taken only when prediction confidence exceeds predefined thresholds.
- A **HOLD zone** is introduced to avoid overtrading.
- Risk is controlled using **position sizing**, **stop‑loss**, and **take‑profit** based on volatility (ATR).

---

## Project Structure
