import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
import pandas as pd
from django.views.decorators.csrf import csrf_exempt

class BTCOHLCVInput(BaseModel):
    """Input schema for BTC OHLCV data retrieval"""
    start_datetime: Optional[str] = Field(None, description="Start datetime in YYYY-MM-DD HH:MM:SS format")
    end_datetime: Optional[str] = Field(None, description="End datetime in YYYY-MM-DD HH:MM:SS format")
    limit: Optional[int] = Field(1000, description="Maximum number of records to return")
    only_non_imputed: Optional[bool] = Field(False, description="Only return non-imputed data")
    interval: Optional[str] = Field(None, description="Optional aggregation interval (e.g., '4H', '1H', '1D'). If provided, aggregates data to this interval using pandas resample. Supported formats: pandas offset aliases like '4H', '1H', '1D', etc.")

class FundingRatesInput(BaseModel):
    """Input schema for funding rates data retrieval"""
    symbol: Optional[str] = Field(None, description="Trading symbol (e.g., 'BTCUSDT')")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    limit: Optional[int] = Field(1000, description="Maximum number of records to return")

class NewsInput(BaseModel):
    """Input schema for news data retrieval"""
    search_term: Optional[str] = Field(None, description="Search term for title or description")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    limit: Optional[int] = Field(100, description="Maximum number of records to return")

class OpenInterestInput(BaseModel):
    """Input schema for open interest data retrieval"""
    start_timestamp: Optional[str] = Field(None, description="Start timestamp in YYYY-MM-DD HH:MM:SS format")
    end_timestamp: Optional[str] = Field(None, description="End timestamp in YYYY-MM-DD HH:MM:SS format")
    limit: Optional[int] = Field(1000, description="Maximum number of records to return")
    interval: Optional[str] = Field(None, description="Optional aggregation interval (e.g., '1H', '4H', '1D'). If provided, aggregates data to this interval using pandas resample. Supported formats: pandas offset aliases like '1h', '4h', '1d', etc.")

class BTCOHLCVTool(BaseTool):
    """Tool for retrieving BTC OHLCV data"""
    name: str = "btc_ohlcv_data"
    description: str = "Retrieve Bitcoin OHLCV (Open, High, Low, Close, Volume) data from the database. Useful for price analysis, technical indicators, and market data queries. Supports optional aggregation interval (e.g., '4h', '1h', '1d')."
    args_schema: type = BTCOHLCVInput
    _db_path: str = PrivateAttr()
    
    def __init__(self, db_path: str):
        super().__init__()
        self._db_path = db_path
    
    def _run(self, start_datetime: Optional[str] = None, end_datetime: Optional[str] = None, 
             limit: int = 1000, only_non_imputed: bool = False, interval: Optional[str] = None) -> str:
        print(f"[BTCOHLCVTool] Called with start_datetime={start_datetime}, end_datetime={end_datetime}, limit={limit}, only_non_imputed={only_non_imputed}, interval={interval}")
        try:
            conn = sqlite3.connect(self._db_path)
            
            query = "SELECT * FROM btc_second_ohlcv WHERE 1=1"
            params = []
            
            if start_datetime:
                query += " AND datetime >= ?"
                params.append(start_datetime)
            
            if end_datetime:
                query += " AND datetime <= ?"
                params.append(end_datetime)
            
            if only_non_imputed:
                query += " AND imputed = 0"
            
            query += " ORDER BY datetime DESC LIMIT ?"
            params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                return "No OHLCV data found for the specified criteria."
            
            # Aggregate if interval is provided
            if interval:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')
                df.set_index('datetime', inplace=True)
                agg_df = df.resample(interval).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                if agg_df.empty:
                    return f"No OHLCV data found after aggregation for interval {interval}."
                # Calculate basic statistics for aggregated data
                stats = {
                    'count': len(agg_df),
                    'price_range': f"{agg_df['low'].min():.2f} - {agg_df['high'].max():.2f}",
                    'avg_volume': f"{agg_df['volume'].mean():.2f}",
                    'latest_close': f"{agg_df.iloc[-1]['close']:.2f}" if not agg_df.empty else "N/A"
                }
                return f"Retrieved {stats['count']} OHLCV records (aggregated by {interval}):\n" + \
                       f"Price Range: ${stats['price_range']}\n" + \
                       f"Average Volume: {stats['avg_volume']}\n" + \
                       f"Latest Close: ${stats['latest_close']}\n\n" + \
                       agg_df.tail(10).to_string()
            # Calculate basic statistics for raw data
            stats = {
                'count': len(df),
                'price_range': f"{df['low'].min():.2f} - {df['high'].max():.2f}",
                'avg_volume': f"{df['volume'].mean():.2f}",
                'latest_close': f"{df.iloc[0]['close']:.2f}" if not df.empty else "N/A"
            }
            
            return f"Retrieved {stats['count']} OHLCV records:\n" + \
                   f"Price Range: ${stats['price_range']}\n" + \
                   f"Average Volume: {stats['avg_volume']}\n" + \
                   f"Latest Close: ${stats['latest_close']}\n\n" + \
                   df.to_string(index=False)
        
        except Exception as e:
            return f"Error retrieving OHLCV data: {str(e)}"

class FundingRatesTool(BaseTool):
    """Tool for retrieving funding rates data"""
    name: str = "funding_rates_data"
    description: str = "Retrieve funding rates data for cryptocurrency perpetual contracts. Useful for analyzing funding costs and market sentiment."
    args_schema: type = FundingRatesInput
    _db_path: str = PrivateAttr()
    
    def __init__(self, db_path: str):
        super().__init__()
        self._db_path = db_path
    
    def _run(self, symbol: Optional[str] = None, start_date: Optional[str] = None, 
             end_date: Optional[str] = None, limit: int = 1000) -> str:
        print(f"[FundingRatesTool] Called with symbol={symbol}, start_date={start_date}, end_date={end_date}, limit={limit}")
        try:
            conn = sqlite3.connect(self._db_path)
            
            query = "SELECT * FROM funding_rates WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if start_date:
                query += " AND date(funding_time) >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date(funding_time) <= ?"
                params.append(end_date)
            
            query += " ORDER BY funding_time DESC LIMIT ?"
            params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                return "No funding rates data found for the specified criteria."
            
            # Calculate statistics
            stats = {
                'count': len(df),
                'avg_funding_rate': f"{df['funding_rate'].mean():.6f}",
                'min_funding_rate': f"{df['funding_rate'].min():.6f}",
                'max_funding_rate': f"{df['funding_rate'].max():.6f}",
                'unique_symbols': df['symbol'].nunique() if 'symbol' in df.columns else 0
            }
            
            return f"Retrieved {stats['count']} funding rate records:\n" + \
                   f"Average Funding Rate: {stats['avg_funding_rate']}\n" + \
                   f"Range: {stats['min_funding_rate']} to {stats['max_funding_rate']}\n" + \
                   f"Unique Symbols: {stats['unique_symbols']}\n\n" + \
                   df.to_string(index=False)
        
        except Exception as e:
            return f"Error retrieving funding rates data: {str(e)}"

class NewsTool(BaseTool):
    """Tool for retrieving news data"""
    name: str = "news_data"
    description: str = "Retrieve cryptocurrency news articles from the database. Useful for sentiment analysis and market context."
    args_schema: type = NewsInput
    _db_path: str = PrivateAttr()
    
    def __init__(self, db_path: str):
        super().__init__()
        self._db_path = db_path
    
    def _run(self, search_term: Optional[str] = None, start_date: Optional[str] = None, 
             end_date: Optional[str] = None, limit: int = 100) -> str:
        print(f"[NewsTool] Called with search_term={search_term}, start_date={start_date}, end_date={end_date}, limit={limit}")
        try:
            conn = sqlite3.connect(self._db_path)
            
            query = "SELECT * FROM news WHERE 1=1"
            params = []
            
            if search_term:
                query += " AND (title LIKE ? OR description LIKE ?)"
                search_pattern = f"%{search_term}%"
                params.extend([search_pattern, search_pattern])
            
            if start_date:
                query += " AND date(date) >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date(date) <= ?"
                params.append(end_date)
            
            query += " ORDER BY date DESC LIMIT ?"
            params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                return "No news articles found for the specified criteria."
            
            # Format news articles for better readability
            formatted_news = []
            for _, row in df.iterrows():
                formatted_news.append(f"ID: {row['id']}\n"
                                    f"Date: {row['date']}\n"
                                    f"Title: {row['title']}\n"
                                    f"Description: {row['description'][:200]}...\n")
            
            return f"Retrieved {len(df)} news articles:\n\n" + "\n".join(formatted_news)
        
        except Exception as e:
            return f"Error retrieving news data: {str(e)}"

class OpenInterestTool(BaseTool):
    """Tool for retrieving open interest data"""
    name: str = "open_interest_data"
    description: str = "Retrieve open interest data for cryptocurrency derivatives. Useful for analyzing market positioning and leverage. Supports optional aggregation interval (e.g., '1H', '4H', '1D')."
    args_schema: type = OpenInterestInput
    _db_path: str = PrivateAttr()
    
    def __init__(self, db_path: str):
        super().__init__()
        self._db_path = db_path
    
    def _run(self, start_timestamp: Optional[str] = None, end_timestamp: Optional[str] = None, 
             limit: int = 1000, interval: Optional[str] = None) -> str:
        print(f"[OpenInterestTool] Called with start_timestamp={start_timestamp}, end_timestamp={end_timestamp}, limit={limit}, interval={interval}")
        try:
            conn = sqlite3.connect(self._db_path)
            
            query = "SELECT * FROM open_interest WHERE 1=1"
            params = []
            
            if start_timestamp:
                query += " AND timestamp >= ?"
                params.append(start_timestamp)
            
            if end_timestamp:
                query += " AND timestamp <= ?"
                params.append(end_timestamp)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                return "No open interest data found for the specified criteria."
            
            # Aggregate if interval is provided
            if interval:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                df.set_index('timestamp', inplace=True)
                agg_df = df.resample(interval).agg({
                    'close_settlement': ['min', 'max', 'mean', 'last'],
                    'close_quote': ['min', 'max', 'mean', 'last']
                }).dropna()
                # Flatten MultiIndex columns for easier access
                agg_df.columns = ['_'.join(col) for col in agg_df.columns]
                print(agg_df)
                if agg_df.empty:
                    return f"No open interest data found after aggregation for interval {interval}."
                # Calculate statistics for aggregated data
                stats = {
                    'count': len(agg_df),
                    'avg_close_settlement': f"{agg_df['close_settlement_mean'].mean():.2f}",
                    'avg_close_quote': f"{agg_df['close_quote_mean'].mean():.2f}",
                    'latest_settlement': f"{agg_df['close_settlement_last'].iloc[-1]:.2f}" if not agg_df.empty else "N/A",
                    'latest_quote': f"{agg_df['close_quote_last'].iloc[-1]:.2f}" if not agg_df.empty else "N/A"
                }
                return f"Retrieved {stats['count']} open interest records (aggregated by {interval}):\n" + \
                       f"Average Close Settlement: {stats['avg_close_settlement']}\n" + \
                       f"Average Close Quote: {stats['avg_close_quote']}\n" + \
                       f"Latest Settlement: {stats['latest_settlement']}\n" + \
                       f"Latest Quote: {stats['latest_quote']}\n\n" + \
                       agg_df.tail(10).to_string()
            # Calculate statistics for raw data
            stats = {
                'count': len(df),
                'avg_close_settlement': f"{df['close_settlement'].mean():.2f}",
                'avg_close_quote': f"{df['close_quote'].mean():.2f}",
                'latest_settlement': f"{df.iloc[0]['close_settlement']:.2f}" if not df.empty else "N/A",
                'latest_quote': f"{df.iloc[0]['close_quote']:.2f}" if not df.empty else "N/A"
            }
            
            return f"Retrieved {stats['count']} open interest records:\n" + \
                   f"Average Close Settlement: {stats['avg_close_settlement']}\n" + \
                   f"Average Close Quote: {stats['avg_close_quote']}\n" + \
                   f"Latest Settlement: {stats['latest_settlement']}\n" + \
                   f"Latest Quote: {stats['latest_quote']}\n\n" + \
                   df.to_string(index=False)
        
        except Exception as e:
            return f"Error retrieving open interest data: {str(e)}"

class DatabaseAnalysisTool(BaseTool):
    """Tool for general database analysis and statistics"""
    name: str = "database_analysis"
    description: str = "Get general statistics and analysis across all database tables. Useful for understanding data availability and coverage."
    _db_path: str = PrivateAttr()
    
    def __init__(self, db_path: str):
        super().__init__()
        self._db_path = db_path
    
    def _run(self, query: str = "") -> str:
        print(f"[DatabaseAnalysisTool] Called with query={query}")
        try:
            conn = sqlite3.connect(self._db_path)
            
            # Get table information
            tables_info = {}
            
            # OHLCV table stats
            ohlcv_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(datetime) as earliest_date,
                    MAX(datetime) as latest_date,
                    SUM(CASE WHEN imputed = 1 THEN 1 ELSE 0 END) as imputed_records
                FROM btc_second_ohlcv
            """, conn)
            tables_info['btc_second_ohlcv'] = ohlcv_stats.to_dict('records')[0]
            
            # Funding rates stats
            funding_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    MIN(funding_time) as earliest_time,
                    MAX(funding_time) as latest_time
                FROM funding_rates
            """, conn)
            tables_info['funding_rates'] = funding_stats.to_dict('records')[0]
            
            # News stats
            news_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date
                FROM news
            """, conn)
            tables_info['news'] = news_stats.to_dict('records')[0]
            
            # Open interest stats
            oi_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as earliest_timestamp,
                    MAX(timestamp) as latest_timestamp
                FROM open_interest
            """, conn)
            tables_info['open_interest'] = oi_stats.to_dict('records')[0]
            
            conn.close()
            
            # Format the results
            result = "Database Analysis Summary:\n\n"
            
            for table_name, stats in tables_info.items():
                result += f"{table_name.upper()}:\n"
                for key, value in stats.items():
                    result += f"  {key}: {value}\n"
                result += "\n"
            
            return result
        
        except Exception as e:
            return f"Error analyzing database: {str(e)}"

# Factory function to create all tools
def create_database_tools(db_path: str) -> List[BaseTool]:
    """Create all database tools for the given database path"""
    return [
        BTCOHLCVTool(db_path),
        FundingRatesTool(db_path),
        NewsTool(db_path),
        OpenInterestTool(db_path),
        DatabaseAnalysisTool(db_path)
    ]

# Example usage:
# tools = create_database_tools("db.sqlite3")
# 
# # Add tools to your LangChain agent
# from langchain.agents import initialize_agent, AgentType
# from langchain.llms import OpenAI
# 
# llm = OpenAI(temperature=0)
# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )