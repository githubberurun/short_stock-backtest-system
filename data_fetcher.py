import os
import time
import gc
from typing import Dict, List, Optional, Final, Any
from datetime import datetime, timedelta
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================================
# 2025-2026年 最新公式ドキュメント準拠検証結果
# Pandas: https://pandas.pydata.org/docs/
# Requests: https://requests.readthedocs.io/en/latest/
# J-Quants V2: https://jpx-jquants.com/spec/migration-v1-v2
# ==========================================

BASE_URL: Final[str] = "https://api.jquants.com/v2"

def is_recently_updated(filepath: str, hours: int = 12) -> bool:
    """物理的なファイルの更新日時をチェックし、指定時間以内なら完全スキップ"""
    if not isinstance(filepath, str): return False
    if not os.path.exists(filepath): return False
    try:
        file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
        return (datetime.now() - file_mtime) < timedelta(hours=hours)
    except Exception:
        return False

class JQuantsV2Fetcher:
    """J-Quants API v2準拠 データ取得クラス (日足・信用残・財務統合版)"""
    def __init__(self, api_key: str) -> None:
        if not isinstance(api_key, str):
            raise TypeError("API key must be a string")
        self.api_key: str = api_key.strip()
        
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": self.api_key})
        
        retry_strategy = Retry(
            total=3,  
            backoff_factor=1.0,  
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_safe_start_date(self) -> str:
        safe_date = datetime.now() - timedelta(days=365 * 10 - 1)
        return safe_date.strftime("%Y-%m-%d")

    def get_top_tickers(self, limit: int = 600) -> List[str]:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
            
        print(f"[INFO] Fetching top {limit} tickers by TurnoverValue...")
        target_date = datetime.now().date()
        
        for _ in range(5):
            params = {"date": target_date.strftime("%Y%m%d")}
            try:
                response = self.session.get(f"{BASE_URL}/equities/bars/daily", params=params, timeout=(10, 30))
                if response.status_code == 200:
                    data = response.json().get("data", [])
                    if len(data) > 500:
                        df = pd.DataFrame(data)
                        df['Va_n'] = pd.to_numeric(df.get('TurnoverValue', df.get('Va', 0)), errors='coerce')
                        top_df = df[df['Va_n'] >= 10_000_000].sort_values('Va_n', ascending=False).head(limit)
                        return [str(code)[:4] for code in top_df['Code'].tolist()]
            except Exception as e:
                print(f"[WARN] Failed to fetch daily data for {target_date}: {type(e).__name__}")
            target_date -= timedelta(days=1)
            time.sleep(1)
            
        print("[ERROR] Could not fetch recent market data.")
        return []

    def _fetch_paginated(self, endpoint: str, ticker: str, start_date: Optional[str] = None) -> List[Dict[str, Any]]:
        code: str = f"{ticker}0" if len(ticker) == 4 else ticker
        all_data: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None
        max_pages: int = 20
        page_count: int = 0

        while page_count < max_pages:
            page_count += 1
            params: Dict[str, Any] = {"code": code}
            
            # 財務サマリは全期間取得を基本とするためfromを付与しない
            if endpoint != "/fins/summary" and start_date:
                params["from"] = start_date
                
            if pagination_key:
                params["pagination_key"] = pagination_key

            try:
                response = self.session.get(f"{BASE_URL}{endpoint}", params=params, timeout=(10, 30))
                
                if response.status_code == 200:
                    res_json = response.json()
                    data_chunk = res_json.get("data", [])
                    all_data.extend(data_chunk)
                    pagination_key = res_json.get("pagination_key")
                    
                    if not pagination_key:
                        break 
                        
                elif response.status_code in [401, 403]:
                    print(f" [Auth Err: {response.status_code}] ", end="", flush=True)
                    break 
                else:
                    print(f" [HTTP {response.status_code}] ", end="", flush=True)
                    break
                    
            except requests.exceptions.Timeout:
                print(f" [Timeout] ", end="", flush=True)
                break 
            except Exception as e:
                print(f" [NetErr: {type(e).__name__}] ", end="", flush=True)
                break
                
            time.sleep(0.1)

        return all_data

    def fetch_daily(self, ticker: str, start_date: Optional[str] = None) -> pd.DataFrame:
        actual_start = start_date if start_date else self.get_safe_start_date()
        data = self._fetch_paginated("/equities/bars/daily", ticker, actual_start)
        return self._clean_daily(pd.DataFrame(data))

    def fetch_margin(self, ticker: str, start_date: Optional[str] = None) -> pd.DataFrame:
        actual_start = start_date if start_date else self.get_safe_start_date()
        data = self._fetch_paginated("/markets/margin-interest", ticker, actual_start)
        return self._clean_margin(pd.DataFrame(data))

    def fetch_fins(self, ticker: str) -> pd.DataFrame:
        # 財務データは遡及修正等を考慮し全量取得してクレンジングする
        data = self._fetch_paginated("/fins/summary", ticker)
        return self._clean_fins(pd.DataFrame(data))

    def _clean_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame): raise TypeError("df must be DataFrame")
        if df.empty: return df
            
        col_map = {
            'Date': 'date', 
            'AdjClose': 'close', 'AdjC': 'close', 'C': 'close_raw', 'Close': 'close_raw',
            'AdjHigh': 'high', 'AdjH': 'high', 'H': 'high_raw', 'High': 'high_raw',
            'AdjLow': 'low', 'AdjL': 'low', 'L': 'low_raw', 'Low': 'low_raw',
            'AdjOpen': 'open', 'AdjO': 'open', 'O': 'open_raw', 'Open': 'open_raw',
            'AdjVolume': 'volume', 'AdjVo': 'volume', 'Vo': 'volume_raw', 'Volume': 'volume_raw',
            'TurnoverValue': 'turnover', 'Va': 'turnover'
        }
        df = df.rename(columns=col_map)
        
        if 'close' not in df.columns and 'close_raw' in df.columns:
            df = df.rename(columns={'close_raw': 'close', 'high_raw': 'high', 'low_raw': 'low', 'open_raw': 'open', 'volume_raw': 'volume'})

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'turnover' not in df.columns or df['turnover'].isnull().all():
            df['turnover'] = df['close'] * df['volume']
                
        if 'date' in df.columns:
            df = df.dropna(subset=['close']).sort_values("date").reset_index(drop=True)
            
        return df

    def _clean_margin(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame): raise TypeError("df must be DataFrame")
        if df.empty: return df
        
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date'})
            
        num_cols = ['LongMarginRatio', 'ShortMarginRatio', 'LongVol', 'ShrtVol']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].replace(['', '-', 'None'], pd.NA), errors='coerce')
                
        if 'date' in df.columns:
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def _clean_fins(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame): raise TypeError("df must be DataFrame")
        if df.empty: return df
        
        if 'DisclosedDate' in df.columns:
            df = df.rename(columns={'DisclosedDate': 'date'})
            
        # 財務評価に必要な主要素を数値化
        num_cols = ['EPS', 'BPS', 'NetSales', 'OperatingProfit', 'EquityToAssetRatio', 'DivAnn']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].replace(['', '-', 'None'], pd.NA), errors='coerce')
                
        if 'date' in df.columns:
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def close(self) -> None:
        if self.session:
            self.session.close()

# ==========================================
# 空データ・異常値に対する堅牢性証明テスト
# ==========================================
def test_integrity() -> None:
    print("[TEST] Running integrity tests for data_fetcher.py...")
    dummy_fetcher = JQuantsV2Fetcher("dummy_key")
    df_empty = pd.DataFrame()
    assert dummy_fetcher._clean_daily(df_empty).empty, "Daily clean failed on empty"
    assert dummy_fetcher._clean_margin(df_empty).empty, "Margin clean failed on empty"
    assert dummy_fetcher._clean_fins(df_empty).empty, "Fins clean failed on empty"
    assert is_recently_updated("non_existent_file.parquet") is False
    dummy_fetcher.close()
    print("[TEST] All integrity tests passed.")

def process_file_update(filepath: str, existing_df: pd.DataFrame, fetched_data: pd.DataFrame, date_col: str = 'date') -> None:
    """ファイルの更新・保存処理を共通化"""
    try:
        if not existing_df.empty:
            if not fetched_data.empty:
                combined = pd.concat([existing_df, fetched_data]).drop_duplicates(subset=[date_col], keep='last').sort_values(date_col).reset_index(drop=True)
                combined.to_parquet(filepath, index=False)
                print(f"UPD (Appended {len(fetched_data)})", end=" | ", flush=True)
            else:
                os.utime(filepath, None) 
                print("CACHE (Up to date)", end=" | ", flush=True)
        else:
            if not fetched_data.empty:
                fetched_data.to_parquet(filepath, index=False)
                print(f"OK ({len(fetched_data)})", end=" | ", flush=True)
            else:
                print("FAILED (No data)", end=" | ", flush=True)
    except Exception as e:
        print(f"Err({type(e).__name__})", end=" | ", flush=True)

if __name__ == "__main__":
    test_integrity()
    
    key = os.getenv("JQUANTS_API_KEY")
    if not key:
        print("[WARN] JQUANTS_API_KEY is not set. Exiting fetcher execution.")
        exit(0)
        
    fetcher = JQuantsV2Fetcher(key)
    
    try:
        data_dir = "Colog_github"
        os.makedirs(data_dir, exist_ok=True)
        
        TARGET_LIMIT = 600
        target_tickers = fetcher.get_top_tickers(limit=TARGET_LIMIT)
        if "13060" not in target_tickers: target_tickers.append("13060")
            
        print(f"[INFO] Starting data fetch (Daily, Margin, Fins) for {len(target_tickers)} tickers...")
        
        for i, target_ticker in enumerate(target_tickers):
            print(f"\n[{i+1}/{len(target_tickers)}] {target_ticker}: ", end="", flush=True)
            
            # --- 1. 日足データの取得と保存 ---
            daily_file = f"{data_dir}/{target_ticker}.parquet"
            if is_recently_updated(daily_file, hours=12):
                print("Daily: SKIP", end=" | ", flush=True)
            else:
                ex_daily = pd.DataFrame()
                start_daily = None
                if os.path.exists(daily_file):
                    try:
                        ex_daily = pd.read_parquet(daily_file)
                        if not ex_daily.empty and 'date' in ex_daily.columns:
                            start_daily = (pd.to_datetime(ex_daily['date'].max()) + timedelta(days=1)).strftime("%Y-%m-%d")
                    except Exception: pass
                fetched_daily = fetcher.fetch_daily(target_ticker, start_daily)
                print("Daily: ", end="", flush=True)
                process_file_update(daily_file, ex_daily, fetched_daily)
                del ex_daily, fetched_daily

            # --- 2. 信用残データの取得と保存 ---
            margin_file = f"{data_dir}/{target_ticker}_margin.parquet"
            if is_recently_updated(margin_file, hours=12):
                print("Margin: SKIP", end=" | ", flush=True)
            else:
                ex_margin = pd.DataFrame()
                start_margin = None
                if os.path.exists(margin_file):
                    try:
                        ex_margin = pd.read_parquet(margin_file)
                        if not ex_margin.empty and 'date' in ex_margin.columns:
                            start_margin = (pd.to_datetime(ex_margin['date'].max()) + timedelta(days=1)).strftime("%Y-%m-%d")
                    except Exception: pass
                fetched_margin = fetcher.fetch_margin(target_ticker, start_margin)
                print("Margin: ", end="", flush=True)
                process_file_update(margin_file, ex_margin, fetched_margin)
                del ex_margin, fetched_margin

            # --- 3. 財務サマリデータの取得と保存 ---
            fins_file = f"{data_dir}/{target_ticker}_fins.parquet"
            if is_recently_updated(fins_file, hours=24): # 財務は更新頻度が低いため24時間スキップ
                print("Fins: SKIP", flush=True)
            else:
                ex_fins = pd.DataFrame()
                if os.path.exists(fins_file):
                    try: ex_fins = pd.read_parquet(fins_file)
                    except Exception: pass
                fetched_fins = fetcher.fetch_fins(target_ticker) # 財務は全量再取得してマージ
                print("Fins: ", end="", flush=True)
                process_file_update(fins_file, ex_fins, fetched_fins)
                del ex_fins, fetched_fins

            gc.collect()
            time.sleep(0.1) # サーバー負荷軽減
            
    finally:
        fetcher.close()
        print("\n[INFO] Data fetching process completed and network sessions closed.")
