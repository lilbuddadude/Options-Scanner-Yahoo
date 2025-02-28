import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import os
import traceback

# Set page config with a title
st.set_page_config(
    page_title="Options Scanner",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Options Arbitrage Scanner")
st.write("Scan for profitable covered call and cash-secured put opportunities")

# Define constants
DB_PATH = 'options_data.db'

# Check if database exists
if os.path.exists(DB_PATH):
    st.success(f"Database found at {DB_PATH}")
else:
    st.error(f"Database not found at {DB_PATH}")

# Add a sidebar
with st.sidebar:
    st.header("Options Scanner")
    
    # Database info
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            st.write(f"Tables in database: {[table[0] for table in tables]}")
            
            # Check data count
            if 'options_data' in [table[0] for table in tables]:
                cursor.execute("SELECT COUNT(*) FROM options_data")
                count = cursor.fetchone()[0]
                st.write(f"Options data records: {count}")
            
            conn.close()
        except Exception as e:
            st.error(f"Error checking database: {str(e)}")
            st.code(traceback.format_exc())
    
    # Refresh button
    if st.button("🔄 Refresh Data from Yahoo Finance", use_container_width=True):
        with st.spinner("Running yahoo_fetch.py..."):
            try:
                import subprocess
                result = subprocess.run(["python", "yahoo_fetch.py"], 
                                       capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    st.success("✅ Data refresh completed successfully")
                    st.code(result.stdout)
                else:
                    st.error(f"❌ Error running yahoo_fetch.py")
                    st.code(result.stderr)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.code(traceback.format_exc())
    
    # Strategy selector
    strategy = st.radio(
        "Strategy",
        ["Covered Calls", "Cash-Secured Puts"],
        index=0
    )
    
    # Scan button
    scan_button = st.button("Scan for Opportunities", type="primary", use_container_width=True)

# Main content area
if scan_button:
    st.write("Scan button clicked!")
    option_type = "covered_call" if strategy == "Covered Calls" else "cash_secured_put"
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Try to get data
        query = f"SELECT * FROM options_data WHERE option_type = '{option_type}' LIMIT 10"
        data = pd.read_sql_query(query, conn)
        
        if not data.empty:
            st.write(f"Found {len(data)} records. Here's a sample:")
            st.dataframe(data)
        else:
            st.warning(f"No data found for {option_type}")
        
        conn.close()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.code(traceback.format_exc())
else:
    # Initial state
    st.info("Use the sidebar to configure and run a scan for option opportunities.")
    
    # Show what the app does
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Covered Call Arbitrage")
        st.write("The scanner looks for covered call opportunities where buying the stock and immediately selling a call option creates an arbitrage situation.")
    
    with col2:
        st.subheader("Cash-Secured Puts")
        st.write("The scanner finds attractive cash-secured put opportunities based on premium, return, and risk metrics.")
