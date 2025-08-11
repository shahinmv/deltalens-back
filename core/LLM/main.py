import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
import requests
from sqlalchemy import create_engine

class BTCOHLCVInput(BaseModel):
    """Input schema for BTC daily OHLCV data retrieval"""
    start_datetime: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format (daily data)")
    end_datetime: Optional[str] = Field(None, description="End date in YYYY-MM-DD format (daily data)")
    limit: Optional[int] = Field(1000, description="Maximum number of records to return")
    only_non_imputed: Optional[bool] = Field(False, description="Only return non-imputed data")
    interval: Optional[str] = Field(None, description="Optional aggregation interval for daily data (e.g., '7D', '30D', '1M'). If provided, aggregates data to this interval using pandas resample. Supported formats: pandas offset aliases like '7D', '30D', '1M', etc.")

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
    description: str = "Retrieve Bitcoin daily OHLCV (Open, High, Low, Close, Volume) data from the database. Contains daily timeframe data useful for price analysis, technical indicators, and market data queries. Supports optional aggregation interval (e.g., '7d', '30d', '1M')."
    args_schema: type = BTCOHLCVInput
    _db_connection_string: str = PrivateAttr()
    
    def __init__(self, db_connection_string: str):
        super().__init__()
        self._db_connection_string = db_connection_string
    
    def _get_sqlalchemy_engine(self):
        """Create SQLAlchemy engine for pandas operations"""
        return create_engine(self._db_connection_string)
    
    def _fetch_today_btc_data(self) -> Optional[pd.DataFrame]:
        """Fetch today's BTC OHLCV data from Binance API"""
        try:
            # Get today's date in UTC
            today = datetime.utcnow().date()
            today_start = datetime.combine(today, datetime.min.time())
            
            # Binance API endpoint for 24hr ticker
            url = "https://api.binance.com/api/v3/ticker/24hr"
            params = {"symbol": "BTCUSDT"}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Create a DataFrame with today's data
            today_data = pd.DataFrame({
                'datetime': [today_start],
                'open': [float(data['openPrice'])],
                'high': [float(data['highPrice'])],
                'low': [float(data['lowPrice'])],
                'close': [float(data['lastPrice'])],
                'volume': [float(data['volume'])]
            })
            
            today_data['datetime'] = pd.to_datetime(today_data['datetime'])
            today_data.set_index('datetime', inplace=True)
            
            return today_data
            
        except Exception as e:
            print(f"[BTCOHLCVTool] Error fetching today's data: {str(e)}")
            return None
    
    def _run(self, start_datetime: Optional[str] = None, end_datetime: Optional[str] = None, 
             limit: int = 1000, only_non_imputed: bool = False, interval: Optional[str] = None) -> str:
        print(f"[BTCOHLCVTool] Called with start_datetime={start_datetime}, end_datetime={end_datetime}, limit={limit}, only_non_imputed={only_non_imputed}, interval={interval}")
        try:
            engine = self._get_sqlalchemy_engine()
            
            query = "SELECT * FROM btc_daily_ohlcv WHERE 1=1"
            params = {}
            
            if start_datetime:
                query += " AND datetime >= %(start_date)s"
                params['start_date'] = start_datetime
            
            if end_datetime:
                query += " AND datetime <= %(end_date)s"
                params['end_date'] = end_datetime
            
            if only_non_imputed:
                query += " AND imputed = 0"
            
            query += " ORDER BY datetime DESC LIMIT %(limit)s"
            params['limit'] = limit
            
            print(f"[BTCOHLCVTool] Executing query: {query}")
            print(f"[BTCOHLCVTool] Query parameters: {params}")
            
            df = pd.read_sql_query(query, engine, params=params)
            
            print(f"[BTCOHLCVTool] Query returned {len(df)} rows")
            if not df.empty:
                print(f"[BTCOHLCVTool] Data preview:")
                print(df.head())
                print(f"[BTCOHLCVTool] Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            else:
                print("[BTCOHLCVTool] No data returned from query")
                
                # Let's check if the table exists and has any data at all
                try:
                    check_query = "SELECT COUNT(*) as total_rows FROM btc_daily_ohlcv"
                    count_df = pd.read_sql_query(check_query, engine)
                    total_rows = count_df.iloc[0]['total_rows']
                    print(f"[BTCOHLCVTool] Total rows in btc_daily_ohlcv table: {total_rows}")
                    
                    if total_rows > 0:
                        # Check date range in the table
                        date_range_query = "SELECT MIN(datetime) as min_date, MAX(datetime) as max_date FROM btc_daily_ohlcv"
                        date_range_df = pd.read_sql_query(date_range_query, engine)
                        print(f"[BTCOHLCVTool] Available date range: {date_range_df.iloc[0]['min_date']} to {date_range_df.iloc[0]['max_date']}")
                except Exception as e:
                    print(f"[BTCOHLCVTool] Error checking table: {str(e)}")
            
            if df.empty:
                return f"No OHLCV data found for the specified criteria. Searched for data between {start_datetime or 'earliest'} and {end_datetime or 'latest'} with limit {limit}. Please check if the btc_daily_ohlcv table contains data for the requested date range."
            
            # Aggregate if interval is provided
            if interval:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')
                df.set_index('datetime', inplace=True)
                
                # Check if we're doing daily aggregation and need today's data
                is_daily_interval = interval.lower() in ['1d', '1day', 'daily', 'd']
                need_today_data = False
                
                if is_daily_interval and not df.empty:
                    # Check if the latest data is from yesterday or earlier
                    latest_date = df.index[-1].date()
                    today = datetime.utcnow().date()
                    yesterday = today - timedelta(days=1)
                    
                    if latest_date <= yesterday:
                        need_today_data = True
                        print(f"[BTCOHLCVTool] Latest data is from {latest_date}, fetching today's data ({today})")
                
                # Fetch and append today's data if needed
                if need_today_data:
                    today_data = self._fetch_today_btc_data()
                    if today_data is not None:
                        # Combine historical and today's data
                        df = pd.concat([df, today_data])
                        df = df.sort_index()
                        print(f"[BTCOHLCVTool] Successfully appended today's data")
                
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
    _db_connection_string: str = PrivateAttr()
    
    def __init__(self, db_connection_string: str):
        super().__init__()
        self._db_connection_string = db_connection_string
    
    def _get_sqlalchemy_engine(self):
        """Create SQLAlchemy engine for pandas operations"""
        return create_engine(self._db_connection_string)
    
    def _run(self, symbol: Optional[str] = None, start_date: Optional[str] = None, 
             end_date: Optional[str] = None, limit: int = 1000) -> str:
        print(f"[FundingRatesTool] Called with symbol={symbol}, start_date={start_date}, end_date={end_date}, limit={limit}")
        try:
            engine = self._get_sqlalchemy_engine()
            
            query = "SELECT * FROM funding_rates WHERE 1=1"
            params = {}
            
            if symbol:
                query += " AND symbol = %(symbol)s"
                params['symbol'] = symbol
            
            if start_date:
                query += " AND date(funding_time) >= %(start_date)s"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND date(funding_time) <= %(end_date)s"
                params['end_date'] = end_date
            
            query += " ORDER BY funding_time DESC LIMIT %(limit)s"
            params['limit'] = limit
            
            df = pd.read_sql_query(query, engine, params=params)
            
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
    _db_connection_string: str = PrivateAttr()
    
    def __init__(self, db_connection_string: str):
        super().__init__()
        self._db_connection_string = db_connection_string
    
    def _get_sqlalchemy_engine(self):
        """Create SQLAlchemy engine for pandas operations"""
        return create_engine(self._db_connection_string)
    
    def _run(self, search_term: Optional[str] = None, start_date: Optional[str] = None, 
             end_date: Optional[str] = None, limit: int = 100) -> str:
        print(f"[NewsTool] Called with search_term={search_term}, start_date={start_date}, end_date={end_date}, limit={limit}")
        try:
            engine = self._get_sqlalchemy_engine()
            
            query = "SELECT * FROM news WHERE 1=1"
            params = {}
            
            if search_term:
                query += " AND (title LIKE %(search_pattern)s OR description LIKE %(search_pattern)s)"
                search_pattern = f"%{search_term}%"
                params['search_pattern'] = search_pattern
            
            if start_date:
                query += " AND date(date) >= %(start_date)s"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND date(date) <= %(end_date)s"
                params['end_date'] = end_date
            
            query += " ORDER BY date DESC LIMIT %(limit)s"
            params['limit'] = limit
            
            df = pd.read_sql_query(query, engine, params=params)
            
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
    _db_connection_string: str = PrivateAttr()
    
    def __init__(self, db_connection_string: str):
        super().__init__()
        self._db_connection_string = db_connection_string
    
    def _get_sqlalchemy_engine(self):
        """Create SQLAlchemy engine for pandas operations"""
        return create_engine(self._db_connection_string)
    
    def _run(self, start_timestamp: Optional[str] = None, end_timestamp: Optional[str] = None, 
             limit: int = 1000, interval: Optional[str] = None) -> str:
        print(f"[OpenInterestTool] Called with start_timestamp={start_timestamp}, end_timestamp={end_timestamp}, limit={limit}, interval={interval}")
        try:
            engine = self._get_sqlalchemy_engine()
            
            query = "SELECT * FROM open_interest WHERE 1=1"
            params = {}
            
            if start_timestamp:
                query += " AND timestamp >= %(start_timestamp)s"
                params['start_timestamp'] = start_timestamp
            
            if end_timestamp:
                query += " AND timestamp <= %(end_timestamp)s"
                params['end_timestamp'] = end_timestamp
            
            query += " ORDER BY timestamp DESC LIMIT %(limit)s"
            params['limit'] = limit
            
            df = pd.read_sql_query(query, engine, params=params)
            
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
    _db_connection_string: str = PrivateAttr()
    
    def __init__(self, db_connection_string: str):
        super().__init__()
        self._db_connection_string = db_connection_string
    
    def _get_sqlalchemy_engine(self):
        """Create SQLAlchemy engine for pandas operations"""
        return create_engine(self._db_connection_string)
    
    def _run(self, query: str = "") -> str:
        print(f"[DatabaseAnalysisTool] Called with query={query}")
        try:
            engine = self._get_sqlalchemy_engine()
            
            # Get table information
            tables_info = {}
            
            # OHLCV table stats
            ohlcv_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(datetime) as earliest_date,
                    MAX(datetime) as latest_date,
                    SUM(CASE WHEN imputed = 1 THEN 1 ELSE 0 END) as imputed_records
                FROM btc_daily_ohlcv
            """, engine)
            tables_info['btc_daily_ohlcv'] = ohlcv_stats.to_dict('records')[0]
            
            # Funding rates stats
            funding_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    MIN(funding_time) as earliest_time,
                    MAX(funding_time) as latest_time
                FROM funding_rates
            """, engine)
            tables_info['funding_rates'] = funding_stats.to_dict('records')[0]
            
            # News stats
            news_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date
                FROM news
            """, engine)
            tables_info['news'] = news_stats.to_dict('records')[0]
            
            # Open interest stats
            oi_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as earliest_timestamp,
                    MAX(timestamp) as latest_timestamp
                FROM open_interest
            """, engine)
            tables_info['open_interest'] = oi_stats.to_dict('records')[0]
            
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
def create_database_tools(db_connection_string: str) -> List[BaseTool]:
    """Create all database tools for the given database connection string"""
    return [
        BTCOHLCVTool(db_connection_string),
        FundingRatesTool(db_connection_string),
        NewsTool(db_connection_string),
        OpenInterestTool(db_connection_string),
        DatabaseAnalysisTool(db_connection_string)
    ]

# Example usage:
# tools = create_database_tools("postgresql://user:password@host:port/database")
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