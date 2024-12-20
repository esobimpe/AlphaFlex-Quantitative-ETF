# 🌟 **AlphaFlex Quantitative Growth ETF (AFG)**  

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Alpha Performance](https://img.shields.io/badge/Alpha-10%25%2B-blue.svg)](https://github.com/AlphaFlexETF/Performance)
[![ETF Type: Growth](https://img.shields.io/badge/ETF%20Type-Growth-yellow.svg)](https://github.com/AlphaFlexETF/Details)

**Link: https://c7e5-136-227-234-163.ngrok-free.app/**

(website is slow and might take a minute to run)

---

> **Disclaimer**: This is a fun project aimed at exploring the concept of an ETF that automatically adjusts itself every week in response to market movements, with the goal of reducing risk and increasing returns. The project leverages advanced quantitative models and strategies, but it is not intended as financial or investment advice. The performance of this ETF is for educational and research purposes only, and any investments based on this project should be made with caution and after thorough research.

---

## 🎯 **Table of Contents**
1. [Overview](#-overview)
2. [Core Features](#-core-features)
3. [Portfolio Construction Methodology](#-portfolio-construction-methodology)
4. [Performance Metrics](#-performance-metrics)
5. [Installation & Usage](#️-installation--usage)
6. [Repository Structure](#-repository-structure)
7. [Why Choose AlphaFlex?](#-why-choose-alphaflex)
8. [Contact Us](#-contact-us)
9. [License](#-license)

---

## 🚀 **Overview**

The **AlphaFlex Quantitative Growth ETF (AFG)** is a next-generation exchange-traded fund designed to capitalize on high-growth opportunities while minimizing risks. By leveraging a unique blend of **fundamental analysis**, **technical indicators**, and **quantitative optimization**, AFG empowers investors to achieve sustainable market-beating returns.

### Key Objectives:
- 📈 **Sustainable Growth**: Identify high-growth companies with robust earnings and sales metrics.
- 🛡️ **Risk Management**: Minimize volatility and drawdowns through disciplined portfolio construction.
- 🌍 **Diversification**: Achieve balanced exposure across sectors and industries for consistent performance.

---

## 🧠 **Core Features**

### 📊 **1. High Earnings Growth**
The High Earnings Growth approach emphasizes identifying companies with consistently robust earnings growth over multiple periods. This method seeks out businesses that demonstrate operational efficiency, strong demand for their products or services, and competitive advantages that enable them to scale profitably. Companies meeting these criteria are often market leaders with clear visibility into their future growth trajectories, making them ideal candidates for inclusion in the AlphaFlex portfolio.

By prioritizing sustained profitability, this screening method ensures the ETF captures firms that can consistently reinvest in innovation and expansion, driving long-term shareholder value. Such businesses typically outperform their peers, not only in bull markets but also during periods of economic uncertainty, due to their resilient financial structures and clear strategic visions.

---

### 📈 **2. High Sales Growth**
High Sales Growth focuses on companies that exhibit significant revenue expansion alongside financial prudence. This approach identifies firms with the ability to scale operations, capture market share, and maintain fiscal discipline, as evidenced by healthy balance sheets and high returns on equity. The methodology also accounts for management quality and their ability to sustain this growth while managing external risks effectively.

This screening process highlights companies that are often innovators or disruptors in their respective industries. By capturing firms with consistent top-line growth, the ETF ensures exposure to businesses with strong underlying demand dynamics, providing a critical foundation for future earnings potential and long-term investment success.

---

### 🤝 **3. Consistent Growth on a Bullish Trend**
The Consistent Growth on a Bullish Trend method leverages both fundamental and technical metrics to identify stocks with a proven track record of earnings expansion and positive market momentum. This dual-focus strategy emphasizes companies that not only deliver strong financial performance but also exhibit a robust upward price trajectory, signaling sustained investor confidence. By blending these factors, the ETF ensures a balanced selection of resilient, well-managed firms that thrive across market cycles.

This method's emphasis on consistency is particularly crucial for mitigating portfolio volatility. Stocks identified through this approach tend to outperform due to their ability to attract long-term institutional investors, reinforcing their price stability. The ETF benefits from aligning with such companies, which have both the operational excellence and market perception to deliver steady growth.

---

### 🛡️ **4. CANSLIM**
Inspired by William O’Neil’s investment framework, the CANSLIM method combines growth, momentum, and operational excellence. This approach identifies companies with a blend of earnings and sales growth, positive market sentiment, and strong technical performance. It also considers management quality and broader industry trends to ensure a comprehensive assessment of each stock's potential.

By leveraging this holistic methodology, AlphaFlex taps into a balanced mix of high-growth opportunities and momentum-driven performance. CANSLIM is particularly effective in capturing stocks that are at the forefront of innovation and market leadership, providing the ETF with exposure to businesses that are well-positioned to benefit from both macroeconomic trends and company-specific drivers.

---

## 📐 Portfolio Construction Methodology

The **AlphaFlex Quantitative Growth ETF (AFG)** is built upon a systematic, data-driven approach to construct a diversified portfolio of high-growth stocks. The portfolio construction methodology leverages various factors including market cap, revenue, volatility, and technical indicators to assign weights to each stock. Below is a detailed explanation of the steps involved in portfolio construction:

### 1. **Stock Selection and Screening**
   - **Country Filter**: The ETF exclusively includes stocks from the United States to ensure regulatory consistency and market familiarity.
   - **Stock Screening**: The stocks are selected based on various quantitative factors, including:
     - High Earnings Growth
     - High Sales Growth
     - Consistent Growth on a Bullish Trend
     - CANSLIM principles (Growth, Momentum, Operational Excellence)

### 2. **Weight Assignment**
   Once stocks are selected for inclusion in the ETF, different methodologies are used to assign a weight to each stock in the portfolio. These weightings are designed to balance between growth potential, risk, and diversification. The weight assignment methods include:

   - **Market Cap Weighting**: Stocks are weighted based on their market capitalization. Larger companies generally receive a larger weight, reflecting their established presence in the market. The market cap weight for each stock is calculated as:
     ```plaintext
     Market Cap Weight = Stock Market Cap / Total Market Cap of Portfolio
     ```

   - **Equal Weighting**: Each stock receives an equal weight in the portfolio, ensuring a broad diversification across all included stocks. The equal weight for each stock is calculated as:
     ```plaintext
     Equal Weight = 1 / Total Number of Stocks
     ```

   - **Fundamental Weighting (Revenue-based)**: Stocks are weighted according to their total revenue, prioritizing companies with higher revenue, which often correlates with stability and growth potential. The weight for each stock based on revenue is:
     ```plaintext
     Fundamental Weight = Stock Revenue / Total Revenue of Portfolio
     ```

   - **Volatility Weighting (Inverse of Volatility)**: To minimize risk, stocks with lower price volatility are given higher weights. This method emphasizes stability by allocating more to stocks with less price fluctuation. The volatility weight for each stock is:
     ```plaintext
     Volatility Weight = (1 / Volatility) / Σ (1 / Volatility of All Stocks)
     ```
     where **Volatility** is calculated as the standard deviation of the stock’s daily closing price over the past year.

   - **Log-Scaled Market Cap Weighting**: For scaling purposes and to avoid disproportionate influence from extremely large companies, the market cap is log-transformed before weighting. This method reduces the impact of exceptionally large companies without excluding them from the portfolio:
     ```plaintext
     Log Market Cap Weight = log(Market Cap + 1) / Σ log(Market Cap of All Stocks + 1)
     ```

### 3. **Final Weight Adjustment**
   After calculating individual weights using the above methods, the final weight for each stock is determined by adjusting and combining the different weightings to achieve a balanced portfolio. The adjusted weight for each stock is computed as a weighted average of the different weight categories:
   ```plaintext
   Adjusted Weight = 0.3 × Log Market Cap Weight + 0.15 × Equal Weight + 0.15 × Volatility Weight + 0.4 × Fundamental Weight
   ```
---


## 📊 **Performance Metrics**

| **Metric**                | **Target**                |
|---------------------------|---------------------------|
| **Annualized Return (CAGR)** | 30%–55%                 |
| **Sharpe Ratio**          | > 1.2                    |
| **Beta**                  | 0.9–1.1                  |
| **Maximum Drawdown**      | -15% to -25%             |

AlphaFlex is designed to deliver **superior long-term returns** while maintaining a moderate risk profile, making it ideal for investors seeking consistent outperformance.

---

## 🌟 Why Choose AlphaFlex? ##
AlphaFlex offers an unparalleled investment opportunity by combining the best practices of quantitative finance with modern risk management.

**📈 Growth Potential**: High-growth companies with proven track records of performance.

**🛡️ Risk Mitigation**: Diversification, beta control, and strict screening rules ensure a balanced approach.

**🌍 Global Reach**: Exposure to a diversified portfolio of global equities.

**🧠 Transparency**: Open-source tools and robust documentation make AlphaFlex a model of investor trust.

AlphaFlex is more than an ETF—it's a comprehensive investment solution designed for outperformance in a dynamic market environment.

## 🛠️ **Installation & Usage**

Clone this repository to access the full AlphaFlex toolkit, including scripts for backtesting, optimization, and analysis.

```bash
git clone https://github.com/esobimpe/AlphaFlex-Quantitative-ETF.git
cd alphaflex-etf



