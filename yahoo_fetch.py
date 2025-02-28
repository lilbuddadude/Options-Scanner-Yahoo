#!/usr/bin/env python3
"""
Options Scanner - Yahoo Finance Data Fetcher
--------------------------------------------
This script fetches options data from Yahoo Finance and saves it to a local SQLite database.
It scans for a much larger universe of stocks to find profitable options opportunities.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import logging
import time
import os
import traceback
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("yahoo_options_fetch.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants
DB_PATH = 'options_data.db'
LOG_PATH = 'yahoo_options_fetch.log'
MAX_WORKERS = 10  # Number of parallel threads for fetching data

# Set to True to use only a small sample of stocks for testing
TEST_MODE = False

def setup_database():
    """Create SQLite database and tables if they don't exist"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS options_data (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            price REAL,
            exp_date TEXT,
            strike REAL,
            option_type TEXT,
            bid REAL,
            ask REAL,
            volume INTEGER,
            open_interest INTEGER,
            implied_volatility REAL,
            delta REAL,
            timestamp DATETIME
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_metadata (
            id INTEGER PRIMARY KEY,
            last_updated DATETIME,
            source TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        logging.info("Database setup complete")
        return True
    except Exception as e:
        logging.error(f"Database setup error: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def fetch_sp500_symbols():
    """Fetch S&P 500 symbols using Wikipedia"""
    try:
        logging.info("Fetching S&P 500 symbols from Wikipedia...")
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_table = tables[0]
        symbols = sp500_table['Symbol'].tolist()
        # Clean symbols - remove dots and handle special cases
        symbols = [symbol.replace('.', '-') for symbol in symbols]
        logging.info(f"Found {len(symbols)} S&P 500 symbols")
        return symbols
    except Exception as e:
        logging.error(f"Error fetching S&P 500 symbols: {str(e)}")
        return []

def fetch_nasdaq100_symbols():
    """Fetch Nasdaq 100 symbols"""
    try:
        logging.info("Fetching Nasdaq 100 symbols from Wikipedia...")
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        nasdaq_table = tables[4]  # Table index may change if Wikipedia page changes
        symbols = nasdaq_table['Ticker'].tolist()
        symbols = [symbol.replace('.', '-') if isinstance(symbol, str) else symbol for symbol in symbols]
        logging.info(f"Found {len(symbols)} Nasdaq 100 symbols")
        return symbols
    except Exception as e:
        logging.error(f"Error fetching Nasdaq 100 symbols: {str(e)}")
        return []

def fetch_high_volume_stocks():
    """Fetch high volume stocks from Yahoo Finance"""
    try:
        logging.info("Fetching high volume stocks...")
        # This is a very basic approximation - in a real app, you'd use a proper stock screener API
        high_volume = ["GME", "AMC", "AAPL", "MSFT", "TSLA", "NVDA", "AMD", "PLTR", "COIN", "RBLX", 
                       "SQ", "PYPL", "LCID", "RIVN", "HOOD", "F", "GM", "BA", "DIS", "NFLX",
                       "META", "AMZN", "GOOG", "BABA", "JD", "PDD", "NIO", "XPEV", "LI", "UBER"]
        return high_volume
    except Exception as e:
        logging.error(f"Error fetching high volume stocks: {str(e)}")
        return []

def get_optionable_stocks():
    """
    Get a list of optionable stocks to scan
    
    Combines S&P 500, Nasdaq 100, and high volume stocks, then filters
    for those with options and within the price range
    """
    if TEST_MODE:
        # Just use a small set for testing
        test_symbols = ["AAPL", "MSFT", "TSLA", "AMD", "NVDA", "PLTR", "COIN", "F", "GM", "T"]
        logging.info(f"TEST MODE: Using {len(test_symbols)} test symbols")
        return test_symbols
    
    # Get symbols from multiple sources
    sp500_symbols = fetch_sp500_symbols()
    nasdaq_symbols = fetch_nasdaq100_symbols()
    volume_symbols = fetch_high_volume_stocks()
    
    # Combine and remove duplicates
    all_symbols = list(set(sp500_symbols + nasdaq_symbols + volume_symbols))
    logging.info(f"Combined list: {len(all_symbols)} unique symbols")
    
    # Filter and validate in batches
    valid_symbols = []
    min_price = 5
    max_price = 100
    
    # Process in batches to avoid rate limiting
    batch_size = 50
    batches = [all_symbols[i:i + batch_size] for i in range(0, len(all_symbols), batch_size)]
    
    logging.info(f"Screening {len(all_symbols)} symbols in {len(batches)} batches...")
    
    for batch_num, symbol_batch in enumerate(batches):
        logging.info(f"Processing batch {batch_num+1}/{len(batches)} ({len(symbol_batch)} symbols)...")
        
        # Process batch with multithreading
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(validate_stock, symbol): symbol for symbol in symbol_batch}
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result:
                        valid_symbols.append(symbol)
                        logging.info(f"Added {symbol} to scan list")
                except Exception as e:
                    logging.error(f"Error validating {symbol}: {str(e)}")
        
        # Prevent rate limiting
        if batch_num < len(batches) - 1:
            logging.info("Pausing between batches to avoid rate limiting...")
            time.sleep(5)
    
    logging.info(f"Found {len(valid_symbols)} valid optionable stocks")
    return valid_symbols

def validate_stock(symbol):
    """Check if a stock meets our criteria for options scanning"""
    try:
        # Get stock info
        stock = yf.Ticker(symbol)
        
        # Check if it has options
        if not stock.options or len(stock.options) == 0:
            return False
        
        # Get price and check range
        info = stock.info
        if 'regularMarketPrice' not in info:
            return False
            
        price = info['regularMarketPrice']
        min_price = 5
        max_price = 100
        
        if price < min_price or price > max_price:
            return False
            
        # Check volume (at least moderate trading activity)
        if 'volume' in info and info['volume'] < 100000:
            return False
            
        return True
    except Exception as e:
        logging.debug(f"Error validating {symbol}: {str(e)}")
        return False

def fetch_options_for_symbol(symbol, option_type="call"):
    """Fetch options chain for a single symbol"""
    try:
        logging.info(f"Fetching {option_type}s for {symbol}...")
        
        # Initialize ticker
        ticker = yf.Ticker(symbol)
        
        # Check if we can get a price
        try:
            current_price = ticker.info.get('regularMarketPrice')
            if not current_price:
                logging.warning(f"Could not get price for {symbol}, skipping")
                return []
        except Exception:
            logging.warning(f"Error getting price for {symbol}, skipping")
            return []
        
        # Get expiration dates
        try:
            expirations = ticker.options
            if not expirations:
                logging.warning(f"No options expiration dates for {symbol}")
                return []
        except Exception:
            logging.warning(f"Error getting options chain for {symbol}")
            return []
        
        # Limit to first few expirations for efficiency
        expirations = expirations[:min(3, len(expirations))]
        
        options_data = []
        
        for expiry in expirations:
            try:
                # Get options chain
                chain = ticker.option_chain(expiry)
                
                # Select calls or puts
                df = getattr(chain, option_type + 's')
                if df.empty:
                    continue
                
                # Format expiration date
                exp_date = datetime.strptime(expiry, "%Y-%m-%d").strftime("%m/%d/%y")
                
                # Process each option
                for _, row in df.iterrows():
                    # Extract basic data
                    strike = float(row['strike'])
                    
                    # Skip very deep ITM or OTM options
                    strike_pct = strike / current_price
                    if option_type == 'call':
                        if strike_pct < 0.7 or strike_pct > 1.3:
                            continue
                    else:  # put
                        if strike_pct < 0.7 or strike_pct > 1.3:
                            continue
                    
                    # Extract option values with fallbacks for nulls
                    bid = float(row['bid']) if row['bid'] > 0 else 0.01
                    ask = float(row['ask']) if row['ask'] > 0 else bid * 1.1
                    volume = int(row['volume']) if not pd.isna(row['volume']) else 0
                    open_interest = int(row['openInterest']) if not pd.isna(row['openInterest']) else 0
                    
                    # Implied volatility (convert to percentage)
                    iv = float(row['impliedVolatility']) * 100 if not pd.isna(row['impliedVolatility']) else 0
                    
                    # Delta (approximate if not available)
                    if 'delta' in row and not pd.isna(row['delta']):
                        delta = float(row['delta'])
                    else:
                        if option_type == 'call':
                            # Higher delta (probability ITM) as strike gets lower relative to price
                            delta = max(0.01, min(0.99, 1.5 - (strike / current_price)))
                        else:  # put
                            # Higher delta (probability ITM) as strike gets higher relative to price
                            delta = max(0.01, min(0.99, (strike / current_price) - 0.5))
                    
                    # Skip if bid or volume is too low
                    if bid < 0.05:
                        continue
                        
                    option_data = {
                        "symbol": symbol,
                        "price": current_price,
                        "exp_date": exp_date,
                        "strike": strike,
                        "bid": bid,
                        "ask": ask,
                        "volume": volume,
                        "open_interest": open_interest,
                        "implied_volatility": iv,
                        "delta": delta,
                    }
                    
                    options_data.append(option_data)
            except Exception as e:
                logging.warning(f"Error processing {symbol} options for {expiry}: {str(e)}")
        
        logging.info(f"Found {len(options_data)} {option_type} options for {symbol}")
        return options_data
    
    except Exception as e:
        logging.error(f"Error fetching options for {symbol}: {str(e)}")
        return []

def fetch_options_data(stock_list, option_type="covered_call"):
    """
    Fetch options data for a list of stocks
    
    Args:
        stock_list: List of stock symbols
        option_type: Either "covered_call" or "cash_secured_put"
    """
    logging.info(f"Fetching {option_type} data for {len(stock_list)} symbols")
    option_chain_type = "call" if option_type == "covered_call" else "put"
    
    all_options = []
    
    # Process stocks with multithreading
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_options_for_symbol, symbol, option_chain_type): symbol 
            for symbol in stock_list
        }
        
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                options = future.result()
                all_options.extend(options)
                logging.info(f"Processed {symbol}: added {len(options)} options")
            except Exception as e:
                logging.error(f"Error processing {symbol}: {str(e)}")
    
    # Convert to DataFrame
    if not all_options:
        logging.warning(f"No {option_type} options data found")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_options)
    
    # Calculate additional metrics
    df = calculate_metrics(df, option_type)
    
    logging.info(f"Finished processing {len(df)} {option_type} options")
    return df

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
    else:  # cash_secured_put
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
    """Save options data to SQLite database"""
    if df.empty:
        logging.warning(f"No {option_type} data to save to database")
        return datetime.now()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear old data of this type
    cursor.execute("DELETE FROM options_data WHERE option_type = ?", (option_type,))
    
    # Insert new data
    timestamp = datetime.now()
    records_inserted = 0
    
    for _, row in df.iterrows():
        cursor.execute('''
        INSERT INTO options_data (
            symbol, price, exp_date, strike, option_type, 
            bid, ask, volume, open_interest, implied_volatility, 
            delta, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['symbol'], row['price'], row['exp_date'], row['strike'], option_type,
            row.get('bid', 0), row.get('ask', 0), row.get('volume', 0), 
            row.get('open_interest', 0), row.get('implied_volatility', 0), row.get('delta', 0), 
            timestamp
        ))
        records_inserted += 1
    
    # Update metadata
    cursor.execute("DELETE FROM data_metadata WHERE source = ?", (option_type,))
    cursor.execute("INSERT INTO data_metadata (last_updated, source) VALUES (?, ?)", 
                 (timestamp, option_type))
    
    conn.commit()
    conn.close()
    
    logging.info(f"Saved {records_inserted} {option_type} records to database")
    return timestamp

def main():
    """Main function to fetch and save options data"""
    try:
        start_time = time.time()
        logging.info("Starting Options Data Fetch from Yahoo Finance")
        
        # Setup database
        setup_database()
        
        # Get stock universe
        stocks = get_optionable_stocks()
        if not stocks:
            logging.error("No valid stocks found to scan")
            return 1
        
        # Fetch covered call data
        logging.info("Fetching covered call data...")
        covered_call_data = fetch_options_data(stocks, "covered_call")
        cc_timestamp = save_to_database(covered_call_data, "covered_call")
        
        # Fetch cash-secured put data
        logging.info("Fetching cash-secured put data...")
        cash_secured_put_data = fetch_options_data(stocks, "cash_secured_put")
        csp_timestamp = save_to_database(cash_secured_put_data, "cash_secured_put")
        
        # Log summary
        elapsed_time = time.time() - start_time
        logging.info(f"Data fetch completed in {elapsed_time:.2f} seconds")
        logging.info(f"Covered Calls: {len(covered_call_data)} records")
        logging.info(f"Cash-Secured Puts: {len(cash_secured_put_data)} records")
        
        return 0
    except Exception as e:
        logging.error(f"Error in data fetch: {str(e)}")
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
