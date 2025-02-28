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
    
    # Expand the list of indices to search
    indices = ['^GSPC', '^NDX', '^DJI']
    
    for index in indices:
        try:
            # Fetch index ticker
            index_ticker = yf.Ticker(index)
            
            # Get tickers for top most liquid stocks
            tickers = ['^AAPL', '^MSFT', '^GOOGL', '^AMZN', '^NVDA', '^META', '^TSLA']
            
            for ticker_symbol in tickers:
                try:
                    # Fetch stock information
                    stock = yf.Ticker(ticker_symbol.replace('^', ''))
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
                        optionable_stocks.append(ticker_symbol.replace('^', ''))
                
                except Exception as e:
                    logging.error(f"Error checking {ticker_symbol}: {e}")
        
        except Exception as e:
            logging.error(f"Error processing index {index}: {e}")
    
    # Add some additional stock screening
    additional_stocks = [
        'AAPL', 'MSFT', 'AMD', 'NVDA', 'GOOGL', 'META', 'INTC', 
        'CSCO', 'QCOM', 'VZ', 'F', 'GM', 'BAC', 'WFC'
    ]
    
    # Combine and deduplicate
    all_stocks = list(set(optionable_stocks + additional_stocks))
    
    # Final filter to ensure stocks meet our criteria
    filtered_stocks = []
    for symbol in all_stocks:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            price = info.get('regularMarketPrice', 0)
            volume = info.get('volume', 0)
            market_cap = info.get('marketCap', 0)
            
            if (min_price <= price <= max_price and 
                volume >= min_volume and 
                market_cap >= min_market_cap and
                stock.options):
                filtered_stocks.append(symbol)
        
        except Exception as e:
            logging.error(f"Error final checking {symbol}: {e}")
    
    logging.info(f"Found {len(filtered_stocks)} optionable stocks")
    return filtered_stocks

def main():
    """Main function to update stock list"""
    try:
        # Get optionable stocks
        optionable_stocks = get_optionable_stocks()
        
        # Update SYMBOLS in the script
        print("Optionable Stocks:", optionable_stocks)
        
        # Optionally, you could write these to a file or database
        # or modify the existing SYMBOLS list
        
        return optionable_stocks
    
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        return []

if __name__ == "__main__":
    main()
