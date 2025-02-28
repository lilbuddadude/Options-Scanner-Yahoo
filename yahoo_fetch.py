#!/usr/bin/env python3
"""
Options Scanner - Yahoo Finance Data Fetcher
--------------------------------------------
This script fetches options data from Yahoo Finance and saves it to a local SQLite database.
It scans for all optionable stocks in the $5-$100 price range to find profitable options opportunities.
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
import re
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
MAX_WORKERS = 4  # Reduced number of parallel threads to avoid rate limiting
MIN_STOCK_PRICE = 5.0  # Minimum stock price to consider
MAX_STOCK_PRICE = 100.0  # Maximum stock price to consider

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

def fetch_russell_2000_symbols():
    """Fetch a subset of Russell 2000 small-cap stocks"""
    # This is a representative sample as there's no easy way to get the full list
    russell_sample = [
        # Financial
        "ABTX", "ACBI", "BANC", "BANF", "BCML", "BHLB", "CADE", "CARE", "CASH", "CATY", "CBSH", "CBU", "CCNE",
        # Technology
        "AAOI", "ACIW", "ACLS", "ACMR", "ACN", "ADBE", "ADI", "ADTN", "AEIS", "AMAT", "AMBA", "AMD", "AMOT",
        # Consumer
        "ANF", "BJRI", "BLMN", "BOOT", "CAKE", "CASY", "CBRL", "CHUY", "COTY", "CROX", "DIN", "DKS", "EAT",
        # Healthcare
        "ACHC", "ACLS", "ADMA", "ADMS", "ADUS", "AERI", "AFMD", "AGEN", "AHCO", "ALEC", "ALKS", "AMPH", "AMRN",
        # Industrial
        "AIMC", "AJRD", "AORT", "APOG", "AVAV", "AXON", "B", "BECN", "BGS", "BKNG", "BLDR", "BWXT", "CAL"
    ]
    logging.info(f"Using {len(russell_sample)} representative Russell 2000 symbols")
    return russell_sample

def fetch_popular_etfs():
    """Fetch a list of popular ETFs with options"""
    etfs = [
        # Major Index ETFs
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", 
        # Sector ETFs
        "XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE",
        # Industry ETFs
        "SMH", "IBB", "XBI", "KRE", "XRT", "XHB", "ITB", "IYT", "XTN",
        # Volatility ETFs
        "VXX", "UVXY", "VIXY",
        # Commodity ETFs
        "GLD", "SLV", "USO", "UNG", "DBC",
        # Bond ETFs
        "TLT", "IEF", "HYG", "LQD", "MUB",
        # Thematic ETFs
        "ARKK", "ARKW", "ARKG", "ARKF", "ARKX", "MSOS", "ROBO", "BOTZ"
    ]
    logging.info(f"Added {len(etfs)} popular ETFs")
    return etfs

def fetch_high_volume_stocks():
    """Fetch high volume stocks"""
    high_volume = [
        # Tech stocks
        "AAPL", "MSFT", "NVDA", "AMD", "INTC", "QCOM", "CSCO", "IBM", "HPQ", "DELL",
        "META", "AMZN", "GOOG", "NFLX", "TSLA", "PLTR", "SNOW", "CRM", "ADBE", "ORCL",
        
        # Finance
        "BAC", "JPM", "WFC", "C", "GS", "MS", "AXP", "V", "MA", "COF",
        
        # Retail
        "WMT", "TGT", "COST", "HD", "LOW", "AMZN", "EBAY", "ETSY", "BBY", "DG",
        
        # Auto
        "F", "GM", "TSLA", "RIVN", "LCID", "TM", "HMC", "STLA", "NIO", "XPEV",
        
        # Energy
        "XOM", "CVX", "BP", "SHEL", "COP", "EOG", "SLB", "HAL", "OXY", "KMI",
        
        # Healthcare
        "JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "AMGN", "GILD", "BIIB", "REGN",
        
        # Consumer goods
        "KO", "PEP", "PG", "CL", "K", "GIS", "CAG", "CPB", "HSY", "KHC",
        
        # Meme/High volatility stocks
        "GME", "AMC", "BB", "BBBY", "WISH", "CLOV", "SPCE", "HOOD", "COIN", "RBLX",
        
        # Cannabis
        "CGC", "TLRY", "ACB", "SNDL", "CRON", "MSOS", "CURLF", "TCNNF",
        
        # Travel & Leisure
        "AAL", "DAL", "UAL", "LUV", "NCLH", "CCL", "RCL", "MAR", "HLT", "H",
        
        # Dividend favorites
        "T", "VZ", "MO", "PM", "BTI", "XOM", "CVX", "KO", "PEP", "JNJ"
    ]
    return high_volume

def fetch_comprehensive_stock_list():
    """Fetch a comprehensive list of stock symbols from multiple sources"""
    all_symbols = set()
    
    # Add from Yahoo Finance screeners
    sources = [
        fetch_sp500_symbols(),       # S&P 500
        fetch_nasdaq100_symbols(),   # NASDAQ 100
        fetch_russell_2000_symbols(),# Russell 2000 (smaller caps)
        fetch_popular_etfs(),        # Popular ETFs
        fetch_high_volume_stocks()   # High volume/popular stocks
    ]
    
    for source in sources:
        all_symbols.update(source)
    
    # Add from predefined lists (industry groups)
    industry_groups = {
        'technology': ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "IBM", "HPQ", "DELL", "CSCO", "ORCL", "CRM", "ADBE", "PLTR"],
        'retail': ["WMT", "TGT", "COST", "HD", "LOW", "AMZN", "EBAY", "ETSY", "BBY", "DG", "DLTR", "KR", "M", "JWN", "GPS"],
        'automotive': ["F", "GM", "TSLA", "RIVN", "LCID", "TM", "HMC", "STLA", "NIO", "XPEV", "LI", "FFIE", "GOEV"],
        'airlines': ["AAL", "DAL", "UAL", "LUV", "SAVE", "JBLU", "ALK", "HA"],
        'cruiselines': ["CCL", "RCL", "NCLH"],
        'entertainment': ["DIS", "NFLX", "PARA", "WBD", "CMCSA"],
        'gaming': ["ATVI", "EA", "TTWO", "RBLX", "U", "CRSR", "HEAR"],
        'social_media': ["META", "SNAP", "PINS", "TWTR", "HOOD", "BMBL"],
        'biotech': ["AMGN", "GILD", "BIIB", "REGN", "MRNA", "BNTX", "NVAX"]
    }
    
    for group, symbols in industry_groups.items():
        all_symbols.update(symbols)
    
    # Remove empty strings and None values
    all_symbols = {s for s in all_symbols if s and isinstance(s, str)}
    
    # Remove symbols with non-standard characters (typically not optionable)
    valid_pattern = re.compile(r'^[A-Z]{1,5}$')
    filtered_symbols = {s for s in all_symbols if valid_pattern.match(s)}
    
    logging.info(f"Built comprehensive list of {len(filtered_symbols)} stock symbols")
    return list(filtered_symbols)

def get_ticker_with_retry(symbol, max_retries=2):
    """Get ticker info with better retry logic and timeout"""
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            
            # Try a minimal info request
            if hasattr(ticker, 'options') and ticker.options:
                return ticker
                
            time.sleep(1)
        except Exception as e:
            logging.debug(f"Attempt {attempt+1} failed for {symbol}: {str(e)}")
            time.sleep(2)
    
    return None

def validate_stock(symbol):
    """Check if a stock has options and is priced $5-$100"""
    try:
        # Get stock info with retry
        stock = get_ticker_with_retry(symbol, max_retries=2)
        if not stock:
            return False
        
        # First quick check - does it have options?
        try:
            options_list = stock.options
            if not options_list or len(options_list) == 0:
                return False
        except:
            return False
        
        # Then get price
        try:
            info = stock.info
            if 'regularMarketPrice' not in info:
                return False
                
            price = info['regularMarketPrice']
            
            if price < MIN_STOCK_PRICE or price > MAX_STOCK_PRICE:
                return False
                
            # Basic volume check - just needs some liquidity
            if 'averageVolume' in info and info['averageVolume'] < 50000:
                return False
                
            logging.info(f"✓ {symbol} validated: price=${price:.2f}, has options")
            return True
        except:
            return False
            
    except Exception as e:
        logging.debug(f"Error validating {symbol}: {str(e)}")
        return False

def ensure_minimum_stocks(valid_symbols):
    """Make sure we have a minimum set of popular optionable stocks"""
    guaranteed_stocks = [
        # Tech giants
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD",
        # Finance
        "JPM", "BAC", "GS", "MS", "V", "MA", "AXP",
        # Consumer
        "WMT", "TGT", "COST", "HD", "MCD", "SBUX", "KO", "PEP",
        # Healthcare
        "JNJ", "PFE", "MRK", "UNH", "CVS",
        # Industrial
        "BA", "CAT", "DE", "GE",
        # Energy
        "XOM", "CVX", "COP", "SLB",
        # Telecom
        "T", "VZ", "TMUS",
        # Media
        "DIS", "NFLX", "CMCSA",
        # Other high-volume
        "F", "GM", "AAL", "CCL", "PLTR", "UBER", "HOOD"
    ]
    
    for symbol in guaranteed_stocks:
        if symbol not in valid_symbols:
            try:
                stock = yf.Ticker(symbol)
                if stock.options and len(stock.options) > 0:
                    info = stock.info
                    if 'regularMarketPrice' in info:
                        price = info['regularMarketPrice']
                        if MIN_STOCK_PRICE <= price <= MAX_STOCK_PRICE:  # Only add if in our price range
                            valid_symbols.append(symbol)
                            logging.info(f"Force-added {symbol} to scan list")
            except Exception:
                pass
    
    return valid_symbols

def get_optionable_stocks():
    """Get all optionable stocks with prices between $5-$100"""
    if TEST_MODE:
        test_symbols = ["AAPL", "MSFT", "TSLA", "AMD", "NVDA", "PLTR", "COIN", "F", "GM", "T"]
        logging.info(f"TEST MODE: Using {len(test_symbols)} test symbols")
        return test_symbols
    
    # Get comprehensive list of stocks to check
    all_symbols = fetch_comprehensive_stock_list()
    
    # Process in small batches with pauses to avoid rate limiting
    batch_size = 20  # Smaller batch size
    batches = [all_symbols[i:i + batch_size] for i in range(0, len(all_symbols), batch_size)]
    
    logging.info(f"Screening {len(all_symbols)} symbols in {len(batches)} batches...")
    
    valid_symbols = []
    for batch_num, symbol_batch in enumerate(batches):
        logging.info(f"Processing batch {batch_num+1}/{len(batches)} ({len(symbol_batch)} symbols)...")
        
        # Use fewer workers to reduce parallel requests
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
        
        # More aggressive pause between batches
        if batch_num < len(batches) - 1:
            pause_time = 10 + (batch_num % 5)  # Variable pause to avoid detection patterns
            logging.info(f"Pausing for {pause_time} seconds after batch {batch_num+1}...")
            time.sleep(pause_time)
            
        # Output progress periodically
        if (batch_num+1) % 10 == 0 or batch_num == len(batches)-1:
            logging.info(f"Progress: Checked {min((batch_num+1)*batch_size, len(all_symbols))} symbols, found {len(valid_symbols)} valid stocks")
    
    # Make sure we have a minimum set of stocks
    valid_symbols = ensure_minimum_stocks(valid_symbols)
    
    logging.info(f"Found {len(valid_symbols)} optionable stocks priced ${MIN_STOCK_PRICE}-${MAX_STOCK_PRICE}")
    
    # Save the list for reference
    try:
        with open('valid_stocks.txt', 'w') as f:
            f.write('\n'.join(valid_symbols))
        logging.info("Saved valid stock list to valid_stocks.txt")
    except Exception as e:
        logging.error(f"Error saving stock list: {str(e)}")
    
    return valid_symbols

def fetch_options_for_symbol(symbol, option_type="call"):
    """Fetch option chain for a single symbol"""
    try:
        logging.info(f"Fetching {option_type}s for {symbol}...")
        
        # Initialize ticker with retry
        ticker = get_ticker_with_retry(symbol)
        if not ticker:
            return []
        
        # Check if we can get a price
        try:
            current_price = ticker.info.get('regularMarketPrice')
            if not current_price or current_price < MIN_STOCK_PRICE or current_price > MAX_STOCK_PRICE:
                logging.warning(f"Price for {symbol} (${current_price}) outside range ${MIN_STOCK_PRICE}-${MAX_STOCK_PRICE}, skipping")
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
                # Get options chain with retry
                chain = None
                for attempt in range(2):
                    try:
                        chain = ticker.option_chain(expiry)
                        break
                    except Exception as e:
                        logging.debug(f"Option chain fetch attempt {attempt+1} failed for {symbol} {expiry}: {str(e)}")
                        if attempt == 1:  # Second attempt failed
                            raise
                        time.sleep(2)
                
                if chain is None:
                    continue
                
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
                    
                    # Focus on options near the money for better trading opportunities
                    strike_pct = strike / current_price
                    if option_type == 'call':
                        # For calls: include strikes from 80% to 120% of current price
                        if strike_pct < 0.8 or strike_pct > 1.2:
                            continue
                    else:  # put
                        # For puts: include strikes from 80% to 100% of current price
                        if strike_pct < 0.8 or strike_pct > 1.0:
                            continue
                    
                    # Extract option values with fallbacks for nulls
                    bid = float(row['bid']) if row['bid'] > 0 else 0.01
                    ask = float(row['ask']) if row['ask'] > 0 else bid * 1.1
                    volume = int(row['volume']) if not pd.isna(row['volume']) else 0
                    open_interest = int(row['openInterest']) if not pd.isna(row['openInterest']) else 0
                    
                    # Skip options with very low liquidity
                    if bid < 0.05:
                        continue
                    
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
    
    # Process in smaller batches to manage memory and avoid timeouts
    batch_size = 10
    stock_batches = [stock_list[i:i + batch_size] for i in range(0, len(stock_list), batch_size)]
    
    for batch_idx, stock_batch in enumerate(stock_batches):
        logging.info(f"Processing {option_type} batch {batch_idx+1}/{len(stock_batches)} ({len(stock_batch)} stocks)")
        
        # Process batch with multithreading
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(fetch_options_for_symbol, symbol, option_chain_type): symbol 
                for symbol in stock_batch
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    options = future.result()
                    all_options.extend(options)
                    logging.info(f"Processed {symbol}: added {len(options)} options")
                except Exception as e:
                    logging.error(f"Error processing {symbol}: {str(e)}")
        
        # Pause between batches
        if batch_idx < len(stock_batches) - 1:
            logging.info("Pausing between batches to avoid rate limiting...")
            time.sleep(5)
            
        # Periodically report progress
        if (batch_idx+1) % 5 == 0 or batch_idx == len(stock_batches)-1:
            logging.info(f"Progress: Processed {min((batch_idx+1)*batch_size, len(stock_list))} stocks, found {len(all_options)} options")
    
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
            records_inserted += 1
            
            # Commit in chunks to avoid huge transactions
            if records_inserted % 1000 == 0:
                conn.commit()
                logging.info(f"Committed {records_inserted} records")
        except Exception as e:
            logging.error(f"Error inserting record: {str(e)}")
    
    # Update metadata
    cursor.execute("DELETE FROM data_metadata WHERE source = ?", (option_type,))
    cursor.execute("INSERT INTO data_metadata (last_updated, source) VALUES (?, ?)", 
                 (timestamp, option_type))
    
    conn.commit()
    conn.close()
    
    logging.info(f"Saved {records_inserted} {option_type} records to database")
    return timestamp

def main():
    """Main function with progress tracking and resume capabilities"""
    try:
        start_time = time.time()
        logging.info("Starting Options Data Fetch from Yahoo Finance")
        
        # Setup database
        setup_database()
        
        # Get all optionable stocks in our price range
        stocks = get_optionable_stocks()
        if not stocks:
            logging.error("No valid stocks found to scan")
            return 1
        
        # Save stock list to file for potential resume
        with open('valid_stocks.txt', 'w') as f:
            f.write('\n'.join(stocks))
        
        # Process options
        logging.info(f"Starting options data fetch for {len(stocks)} stocks")
        
        # Process covered calls
        logging.info("Fetching covered call data...")
        covered_call_data = fetch_options_data(stocks, "covered_call")
        cc_timestamp = save_to_database(covered_call_data, "covered_call")
        
        # Process puts
        logging.info("Fetching cash-secured put data...")
        cash_secured_put_data = fetch_options_data(stocks, "cash_secured_put")
        csp_timestamp = save_to_database(cash_secured_put_data, "cash_secured_put")
        
        # Log summary
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Create a summary
        symbols_with_calls = covered_call_data['symbol'].nunique() if not covered_call_data.empty else 0
        symbols_with_puts = cash_secured_put_data['symbol'].nunique() if not cash_secured_put_data.empty else 0
        
        summary = [
            f"Data fetch completed in {time_str}",
            f"Scanned {len(stocks)} stocks priced between ${MIN_STOCK_PRICE}-${MAX_STOCK_PRICE}",
            f"Covered Calls: {len(covered_call_data)} records across {symbols_with_calls} stocks",
            f"Cash-Secured Puts: {len(cash_secured_put_data)} records across {symbols_with_puts} stocks",
            f"Total Options: {len(covered_call_data) + len(cash_secured_put_data)}"
        ]
        
        for line in summary:
            logging.info(line)
        
        return 0
    except Exception as e:
        logging.error(f"Error in data fetch: {str(e)}")
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
