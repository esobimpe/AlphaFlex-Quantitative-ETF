from flask import Flask, render_template
import os
import yfinance as yf
import numpy as np
import pandas as pd

# Set Flask app and specify template folder as current directory
app = Flask(__name__, template_folder=os.getcwd())

# Function for sector allocation
def calculate_sector_allocation(df):
    sector_allocation = df.groupby('Sector')['Weights'].sum()
    return sector_allocation.sort_values(ascending=False)

# Main function to fetch stock data and calculate weights
def get_stock_data():

    #Stocks are manually entered in the meantime. The code will be updated to automatically fetch stocks that meet our features in few weeks
    stock_tickers = [
        "WGS", "APP", "CRDO", "GRND", "TKO", "ALKT", "ATAT", "YMM", "SOFI", "GLBE", 
        "AMBA", "RELY", "AXON", "MPWR", "NBIX", "NOW", "NVDA", "SHOP", "NFLX", 
        "SHC", "ADMA", "CDXC", "CLSK", "DOCS", "FOLD", "FUTU", "MNKD", "ORC", 
        "QUBT", "RERE", "SCPX", "SMCI", "TALK", "VXRT"
    ]

    data = []
    for ticker in stock_tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        data.append({
            'Stock': ticker,
            'Name': info.get('shortName', 'N/A'),
            'Country': info.get('country', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Market Cap': info.get('marketCap', 0),
            'Revenue': info.get('totalRevenue', 0),
            'Volatility': np.std(hist['Close'])
        })

    df = pd.DataFrame(data)
    df = df[df['Country'].isin(['United States', 'Canada'])]
    total_market_cap = df['Market Cap'].sum()
    df['Market Cap Weight'] = df['Market Cap'] / total_market_cap
    num_stocks = len(stock_tickers)
    df['Equal Weight'] = 1 / num_stocks
    total_revenue = df['Revenue'].sum()
    df['Fundamental Weight'] = df['Revenue'] / total_revenue
    df['Inverse Volatility'] = 1 / df['Volatility']
    total_inverse_vol = df['Inverse Volatility'].sum()
    df['Volatility Weight'] = df['Inverse Volatility'] / total_inverse_vol
    df['Log Market Cap'] = np.log(df['Market Cap'] + 1)
    df['Log Market Cap Weight'] = df['Log Market Cap'] / df['Log Market Cap'].sum()
    total_capped_weight = df['Log Market Cap Weight'].sum()
    df['Final Capped Weight'] = (df['Log Market Cap Weight'] / total_capped_weight) * 100

    df['Adjusted Weight'] = (
        df['Final Capped Weight'] * 0.3 +
        df['Equal Weight'] * 0.15 +
        df['Volatility Weight'] * 0.15 +
        df['Fundamental Weight'] * 0.4
    )

    total_adjusted_weight = df['Adjusted Weight'].sum()
    df['Final Adjusted Weight %'] = (df['Adjusted Weight'] / total_adjusted_weight) * 100

    final_df = df[['Stock', 'Name', 'Country', 'Sector', 'Market Cap', 'Revenue', 'Volatility', 'Final Adjusted Weight %']]
    return final_df.sort_values(by='Final Adjusted Weight %', ascending=False)

def consolidated_portfolio_values_with_totals(stock_weights, initial_investment=10000):
    periods = {"1d": "1d", "5d": "5d", "1mo": "1mo", "3mo": "3mo", "6mo": "6mo", "1y": "1y", "2y": "2y", "5y": "5y"}
    value_data = {'Ticker': stock_weights['Stock'], 'Name': stock_weights['Name'], 'Sector': stock_weights['Sector'],
                  'Market Cap': stock_weights['Market Cap'], 'Revenue': stock_weights['Revenue'], 
                  'Volatility': stock_weights['Volatility'], 'Weights': stock_weights['Final Adjusted Weight %']}

    for label, period in periods.items():
        stock_prices = {}
        for stock in stock_weights['Stock']:
            try:
                stock_data = yf.Ticker(stock).history(period=period)['Close']
                stock_prices[stock] = stock_data
            except Exception as e:
                stock_prices[stock] = None

        price_df = pd.DataFrame(stock_prices).dropna(axis=1)
        if price_df.empty:
            value_data[f'{label} Value'] = [np.nan] * len(stock_weights)
            continue

        stock_weights['Normalized Weight'] = stock_weights['Final Adjusted Weight %'] / stock_weights['Final Adjusted Weight %'].sum()
        weights = stock_weights.set_index('Stock')['Normalized Weight']
        weights = weights.reindex(price_df.columns).fillna(0)
        initial_stock_values = weights * initial_investment
        price_change_ratios = price_df.iloc[-1] / price_df.iloc[0]
        final_stock_values = initial_stock_values * price_change_ratios
        value_data[f'{label} Value'] = final_stock_values.reindex(stock_weights['Stock']).fillna(0).values

    value_data['Initial Value'] = (stock_weights['Final Adjusted Weight %'] / 100) * initial_investment
    final_df = pd.DataFrame(value_data)
    total_row = {
        'Ticker': 'TOTAL',
        'Name': '',
        'Weights': final_df['Weights'].sum(),
        'Initial Value': final_df['Initial Value'].sum()
    }
    for label in periods.keys():
        total_row[f'{label} Value'] = final_df[f'{label} Value'].sum()
    final_df = pd.concat([final_df, pd.DataFrame([total_row])], ignore_index=True)
    return final_df

@app.route('/')
def index():
    stock_weights = get_stock_data()
    consolidated_df = consolidated_portfolio_values_with_totals(stock_weights)
    sector_allocation = calculate_sector_allocation(consolidated_df)
    total_returns = consolidated_df.iloc[-1, -8:-1].to_dict()  # Total returns for 1d to 5y

    return render_template(
        'index.html',
        table_data=consolidated_df.to_dict('records'),
        sector_allocation=sector_allocation.to_dict(),
        total_returns=total_returns
    )

if __name__ == '__main__':
    app.run(debug=True)
