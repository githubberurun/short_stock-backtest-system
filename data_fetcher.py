import os
import requests
import pandas as pd
import time
from typing import Dict, List, Optional, Final, Any
from datetime import datetime, timedelta

# ==========================================
# 2025-2026年 最新公式ドキュメント準拠
# Pandas: https://pandas.pydata.org/docs/
# Requests: https://requests.readthedocs.io/en/latest/
# ==========================================

BASE_URL: Final[str] = "https://api.jquants.com/v2"
PRICES_ENDPOINT: Final[str] = "/equities/bars/daily"
INFO_ENDPOINT: Final[str] = "/equities/master" 

class JQuantsV2Fetcher:
    """J-Quants API v2準拠 生存バイアス排除・全小型株データ取得クラス"""
    def __init__(self, api_key: str) -> None:
        if not isinstance(api_key, str):
            raise TypeError("API key must be a string")
        self.api_key: str = api_key.strip()
        self.headers: Dict[str, str] = {"x-api-key": self.api_key}

    def get_safe_start_date(self) -> str:
        safe_date = datetime.now() - timedelta(days=365 * 10 - 1)
        return safe_date.strftime("%Y-%m-%d")

    def get_all_small_cap_tickers(self) -> List[str]:
        """現在時点の流動性で絞り込まず、対象市場の全銘柄を抽出する"""
        print("[INFO] Fetching master listed info to get all candidates...", flush=True)
        candidate_codes: List[str] = []
        
        try:
            info_resp = requests.get(f"{BASE_URL}{INFO_ENDPOINT}", headers=self.headers, timeout=30)
            if info_resp.status_code == 200:
                info_data = info_resp.json().get("data", [])
                if info_data:
                    info_df = pd.DataFrame(info_data)
                    
                    market_col = next((c for c in ["MarketCodeName", "MktNm", "Section", "Segment"] if c in info_df.columns), None)
                    sector_col = next((c for c in ["SectorName", "S33Nm", "S17Nm"] if c in info_df.columns), None)
                    code_col = next((c for c in ["Code", "code"] if c in info_df.columns), "Code")
                    
                    if market_col and code_col in info_df.columns:
                        small_cap_segments = ["Growth", "Standard", "グロース", "スタンダード", "G", "S"]
                        candidates_info = info_df[info_df[market_col].astype(str).str.contains("|".join(small_cap_segments), na=False, case=False)]
                        
                        if sector_col:
                            candidates_info = candidates_info[~candidates_info[sector_col].astype(str).str.contains("ETF|ETN|REIT", na=False, case=False)]
                            
                        candidate_codes = [str(c)[:4] for c in candidates_info[code_col].tolist()]
                        print(f"[INFO] Found {len(candidate_codes)} tickers from Growth/Standard segments.", flush=True)
                    else:
                        print(f"[WARN] Valid Market column not found.", flush=True)
            else:
                print(f"[WARN] Master info fetch failed ({info_resp.status_code}).", flush=True)
        except Exception as e:
            print(f"[WARN] Exception in master info fetch: {e}.", flush=True)

        return list(set(candidate_codes))

    def fetch(self, ticker: str) -> pd.DataFrame:
        if not isinstance(ticker, str):
            raise TypeError("ticker must be a string")
            
        code: str = f"{ticker}0" if len(ticker) == 4 else ticker
        start_date: str = self.get_safe_start_date()
        all_data: List[Dict[str, Any]] = []
        pagination_key: Optional[str] = None

        while True:
            params: Dict[str, Any] = {"code": code, "from": start_date}
            if pagination_key:
                params["pagination_key"] = pagination_key

            try:
                response = requests.get(f"{BASE_URL}{PRICES_ENDPOINT}", headers=self.headers, params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Network error: {e}")
                return pd.DataFrame()

            if response.status_code != 200:
                print(f"[ERROR] API {response.status_code}: {response.text}")
                return pd.DataFrame()

            res_json = response.json()
            quotes = res_json.get("data", res_json.get("daily_quotes", []))
            all_data.extend(quotes)

            pagination_key = res_json.get("pagination_key")
            if not pagination_key:
                break
            time.sleep(0.1)

        return self._clean(pd.DataFrame(all_data))

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame): raise TypeError("df must be a pandas DataFrame")
        if df.empty: return df
            
        col_map = {
            'Date': 'date', 
            'AdjClose': 'close', 'AdjC': 'close', 'AdjustmentClose': 'close', 'C': 'close_raw', 'Close': 'close_raw',
            'AdjHigh': 'high', 'AdjH': 'high', 'AdjustmentHigh': 'high', 'H': 'high_raw', 'High': 'high_raw',
            'AdjLow': 'low', 'AdjL': 'low', 'AdjustmentLow': 'low', 'L': 'low_raw', 'Low': 'low_raw',
            'AdjOpen': 'open', 'AdjO': 'open', 'AdjustmentOpen': 'open', 'O': 'open_raw', 'Open': 'open_raw',
            'AdjVolume': 'volume', 'AdjVo': 'volume', 'AdjustmentVolume': 'volume', 'Vo': 'volume_raw', 'Volume': 'volume_raw',
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

def test_integrity() -> None:
    print("[TEST] Running integrity tests for data_fetcher.py...")
    dummy_fetcher = JQuantsV2Fetcher("dummy_key")
    df_empty = pd.DataFrame()
    assert dummy_fetcher._clean(df_empty).empty, "Empty DataFrame should return empty DataFrame"
    print("[TEST] All integrity tests passed.")

if __name__ == "__main__":
    test_integrity()
    
    key = os.getenv("JQUANTS_API_KEY")
    if not key:
        print("[WARN] JQUANTS_API_KEY is not set. Exiting fetcher execution.")
        exit(0)
        
    fetcher = JQuantsV2Fetcher(key)
    os.makedirs("Colog_github", exist_ok=True)
    
    target_tickers = fetcher.get_all_small_cap_tickers()
    print(f"\n[INFO] Starting data fetch for {len(target_tickers)} tickers...", flush=True)
    
    for i, target_ticker in enumerate(target_tickers):
        path = f"Colog_github/{target_ticker}.parquet"
        
        if os.path.exists(path):
            print(f"[{i+1}/{len(target_tickers)}] Fetching {target_ticker}... CACHED (SKIP)", flush=True)
            continue
            
        print(f"[{i+1}/{len(target_tickers)}] Fetching {target_ticker}...", end=" ", flush=True)
        fetched_data = fetcher.fetch(target_ticker)
        if not fetched_data.empty:
            fetched_data.to_parquet(path, index=False)
            print(f"OK ({len(fetched_data)} rows)", flush=True)
        else:
            print("FAILED", flush=True)
