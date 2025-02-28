import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import logging
import time
import os
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database path
DB_PATH = 'options_data.db'

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
        logging.info("Database setup completed successfully")
        return True
    except Exception as e:
        logging.error(f"Database setup error: {str(e)}")
        return False

def get_optionable_stocks(
    min_price=5, 
    max_price=1000,  # Increased max price to capture high-priced stocks
    min_volume=50000,  # Lowered volume requirement
    min_market_cap=100_000_000  # Lowered market cap requirement
):
    """
    Fetch a comprehensive list of optionable stocks within specified parameters
    """
    optionable_stocks = []
    
    # Expanded list of stocks to check
    stocks_to_check = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 
        'CSCO', 'QCOM', 'F', 'GM', 'BAC', 'WFC', 'JPM', 'XOM', 'CVX', 'PFE',
        'MRK', 'JNJ', 'UNH', 'HD', 'WMT', 'TGT', 'COST', 'DIS', 'NFLX', 'SMCI',
        'PYPL', 'SQ', 'COIN', 'SPCE', 'GME', 'AMC', 'PLTR', 'RBLX', 'U', 'ABNB'
    ]
    
    for symbol in stocks_to_check:
        try:
            logging.info(f"Checking {symbol} for options...")
            # Fetch stock information
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Check if we got valid info
            if not info or 'regularMarketPrice' not in info:
                logging.warning(f"Could not get valid info for {symbol}, skipping")
                continue
                
            # Extract basic metrics
            price = info.get('regularMarketPrice', 0)
            volume = info.get('volume', 0)
            market_cap = info.get('marketCap', 0)
            
            # Check if stock has options
            has_options = bool(stock.options)
            
            logging.info(f"{symbol}: Price=${price}, Volume={volume}, MarketCap={market_cap}, HasOptions={has_options}")
            
            # Check if stock meets our criteria
            checks = (
                has_options and  # Has options chain
                min_price <= price <= max_price and
                volume >= min_volume and
                market_cap >= min_market_cap
            )
            
            if checks:
                optionable_stocks.append(symbol)
                logging.info(f"Added {symbol} to optionable stocks list")
        
        except Exception as e:
            logging.error(f"Error checking {symbol}: {e}")
            logging.debug(traceback.format_exc())
    
    logging.info(f"Found {len(optionable_stocks)} optionable stocks: {optionable_stocks}")
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
    
    logging.info(f"Fetching {option_type} data for {len(symbols)} symbols")
    
    all_options = []
    option_chain_type = "calls" if option_type == "covered_call" else "puts"
    
    for symbol in symbols:
        try:
            logging.info(f"Processing {symbol}...")
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
                
            logging.info(f"{symbol} has {len(expirations)} expiration dates")
            
            # Use first 3 expiration dates or fewer if less are available
            for date in expirations[:min(3, len(expirations))]:
                try:
                    logging.info(f"Getting {symbol} options for {date}")
                    # Get options chain for date
                    chain = ticker.option_chain(date)
                    
                    # Select calls or puts
                    df = getattr(chain, option_chain_type)
                    
                    if df.empty:
                        logging.warning(f"No {option_chain_type} data for {symbol} on {date}")
                        continue
                        
                    logging.info(f"Found {len(df)} {option_chain_type} for {symbol} on {date}")
                    
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
                            "option_type": option_type,
                            "bid": bid,
                            "ask": ask,
                            "volume": volume,
                            "open_interest": oi,
                            "implied_volatility": iv,
                            "delta": delta,
                            "timestamp": datetime.now()
                        }
                        
                        all_options.append(option_data)
                
                except Exception as e:
                    logging.error(f"Error processing {symbol} {date}: {str(e)}")
                    logging.debug(traceback.format_exc())
        
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
            logging.debug(traceback.format_exc())
    
    return pd.DataFrame(all_options)

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
        try:
            cursor.execute('''
            INSERT INTO options_data (
                symbol, price, exp_date, strike, option_type, 
                bid, ask, volume, open_interest, implied_volatility, 
                delta, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['symbol'], 
                row['price'], 
                row['exp_date'], 
                row['strike'], 
                option_type,
                row['bid'], 
                row['ask'], 
                row['volume'], 
                row['open_interest'], 
                row['implied_volatility'],
                row['delta'], 
                timestamp
            ))
            count += 1
        except Exception as e:
            logging.error(f"Error inserting record: {e}")
            logging.debug(f"Problematic row: {row}")
    
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
        # Setup database first
        setup_database()
        
        # Fetch optionable stocks
        optionable_stocks = get_optionable_stocks()
        
        if not optionable_stocks:
            logging.error("No optionable stocks found, cannot proceed")
            return 1
            
        # Process covered calls
        logging.info("Fetching covered call data...")
        cc_data = fetch_options_data("covered_call", optionable_stocks)
        if not cc_data.empty:
            save_to_database(cc_data, "covered_call")
        else:
            logging.warning("No covered call data found")
        
        # Process cash secured puts
        logging.info("Fetching cash secured put data...")
        csp_data = fetch_options_data("cash_secured_put", optionable_stocks)
        if not csp_data.empty:
            save_to_database(csp_data, "cash_secured_put")
        else:
            logging.warning("No cash secured put data found")
        
        logging.info("Data refresh complete!")
        
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        logging.debug(traceback.format_exc())
        return 1
        
    return 0

if __name__ == "__main__":
    main()
