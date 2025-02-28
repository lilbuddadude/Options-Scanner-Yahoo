import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import logging
import time
import os

# Setup logging
logging.basicConfig(level=logging.INFO)

# Database path
DB_PATH = 'options_data.db'

def get_optionable_stocks(
    min_price=5, 
    max_price=100, 
    min_volume=100000, 
    min_market_cap=500_000_000
):
    """
    Fetch a comprehensive list of optionable stocks within specified parameters
    """
    optionable_stocks = []
    
    # Predefined list of stocks to check
    stocks_to_check = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 
        'AMD', 'INTC', 'CSCO', 'QCOM', 'F', 'GM', 'BAC', 'WFC'
    ]
    
    for symbol in stocks_to_check:
        try:
            # Fetch stock information
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Comprehensive checks for stock selection
            price = info.get('regularMarketPrice', 0)
            volume = info.get('volume', 0)
            market_cap = info.get('marketCap', 0)
            
            # Check if stock meets our criteria
            checks = (
                stock.options and  # Has options chain
                min_price <= price <= max_price and
                volume >= min_volume and
                market_cap >= min_market_cap
            )
            
            if checks:
                optionable_stocks.append(symbol)
        
        except Exception as e:
            logging.error(f"Error checking {symbol}: {e}")
    
    logging.info(f"Found {len(optionable_stocks)} optionable stocks")
    return optionable_stocks

def fetch_options_data(option_type="covered_call", symbols=None):
    """
    Fetch options data from Yahoo Finance
    option_type: either "covered_call" or "cash_secured_put"
    symbols: list of symbols to fetch (if None, use predefined list)
    """
    # Use provided symbols or get optionable stocks
    if symbols is None:
        symbols = get_optionable_stocks()
    
    logging.info(f"Fetching {option_type} data for {symbols}")
    
    all_options = []
    option_chain_type = "calls" if option_type == "covered_call" else "puts"
    
    for symbol in symbols:
        try:
            # Get ticker info
            ticker = yf.Ticker(symbol)
            current_price = ticker.info.get('regularMarketPrice')
            if not current_price:
                logging.warning(f"Could not get price for {symbol}, skipping")
                continue
                
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                logging.warning(f"No options data for {symbol}, skipping")
                continue
                
            # Use first 3 expiration dates
            for date in expirations[:3]:
                try:
                    # Get options chain for date
                    chain = ticker.option_chain(date)
                    
                    # Select calls or puts
                    df = getattr(chain, option_chain_type)
                    
                    # Process each option
                    for _, row in df.iterrows():
                        # Format date
                        exp_date = datetime.strptime(date, "%Y-%m-%d").strftime("%m/%d/%y")
                        
                        # Calculate needed values
                        strike = float(row['strike'])
                        bid = float(row['bid']) if row['bid'] > 0 else 0.01
                        ask = float(row['ask']) if row['ask'] > 0 else bid * 1.1
                        volume = int(row['volume']) if not pd.isna(row['volume']) else 0
                        oi = int(row['openInterest']) if not pd.isna(row['openInterest']) else 0
                        iv = float(row['impliedVolatility']) * 100 if not pd.isna(row['impliedVolatility']) else 30
                        
                        # Calculate delta if not available
                        if 'delta' in row and not pd.isna(row['delta']):
                            delta = float(row['delta'])
                        else:
                            # Approximate delta based on strike relation to price
                            if option_type == "covered_call":
                                delta = max(0.1, min(0.99, 1 - (strike / current_price) * 1.1))
                            else:
                                delta = max(0.1, min(0.99, (strike / current_price) * 0.8))
                        
                        option_data = {
                            "symbol": symbol,
                            "price": current_price,
                            "exp_date": exp_date,
                            "strike": strike,
                            "bid": bid,
                            "ask": ask,
                            "volume": volume,
                            "open_int": oi,
                            "iv_pct": iv,
                            "delta": delta,
                            "option_type": option_type
                        }
                        
                        all_options.append(option_data)
                
                except Exception as e:
                    logging.error(f"Error processing {symbol} {date}: {str(e)}")
        
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
    
    return pd.DataFrame(all_options)

def calculate_metrics(df, option_type):
    """Calculate trading metrics based on option type"""
    if df.empty:
        return df
        
    # Calculate days to expiry
    df['days_to_expiry'] = df['exp_date'].apply(
        lambda x: max(1, (datetime.strptime(x, "%m/%d/%y") - datetime.now()).days)
    )
    
    if option_type == "covered_call":
        # Calculate covered call metrics
        df['moneyness'] = (df['strike'] - df['price']) / df['price'] * 100
        df['net_profit'] = (df['bid'] * 100) - ((100 * df['price']) - (100 * df['strike']))
        df['be_bid'] = df['price'] - df['bid']
        df['be_pct'] = (df['be_bid'] - df['price']) / df['price'] * 100
        df['otm_prob'] = (1 - df['delta']) * 100
    else:
        # Calculate cash-secured put metrics
        df['moneyness'] = (df['strike'] - df['price']) / df['price'] * 100
        df['net_profit'] = (df['bid'] * 100) - ((100 * df['strike']) - (100 * df['price']))
        df['be_bid'] = df['strike'] - df['bid']
        df['be_pct'] = (df['be_bid'] - df['price']) / df['price'] * 100
        df['otm_prob'] = df['delta'] * 100
    
    # Calculate returns
    df['pnl_rtn'] = (df['bid'] / df['price']) * 100
    df['ann_rtn'] = df['pnl_rtn'] * (365 / df['days_to_expiry'])
    
    return df

def save_to_database(df, option_type):
    """Save options data to database"""
    if df.empty:
        logging.warning(f"No {option_type} data to save")
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear existing data of this type
    cursor.execute("DELETE FROM options_data WHERE option_type = ?", (option_type,))
    
    # Insert new data
    timestamp = datetime.now()
    count = 0
    
    for _, row in df.iterrows():
        cursor.execute('''
        INSERT INTO options_data (
            symbol, price, exp_date, strike, option_type, 
            bid, ask, volume, open_interest, implied_volatility, 
            delta, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['symbol'], row['price'], row['exp_date'], row['strike'], option_type,
            row['bid'], row['ask'], row['volume'], row['open_int'], row['iv_pct'],
            row['delta'], timestamp
        ))
        count += 1
    
    # Update metadata
    cursor.execute("DELETE FROM data_metadata WHERE source = ?", (option_type,))
    cursor.execute("INSERT INTO data_metadata (last_updated, source) VALUES (?, ?)", 
                  (timestamp, option_type))
    
    conn.commit()
    conn.close()
    
    logging.info(f"Saved {count} {option_type} records to database")

def main():
    """Main function to fetch and save options data"""
    try:
        # Fetch optionable stocks
        optionable_stocks = get_optionable_stocks()
        
        # Process covered calls
        logging.info("Fetching covered call data...")
        cc_data = fetch_options_data("covered_call", optionable_stocks)
        cc_data = calculate_metrics(cc_data, "covered_call")
        save_to_database(cc_data, "covered_call")
        
        # Process cash secured puts
        logging.info("Fetching cash secured put data...")
        csp_data = fetch_options_data("cash_secured_put", optionable_stocks)
        csp_data = calculate_metrics(csp_data, "cash_secured_put")
        save_to_database(csp_data, "cash_secured_put")
        
        logging.info("Data refresh complete!")
        
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    main()
