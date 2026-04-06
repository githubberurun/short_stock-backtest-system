import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Final
from datetime import datetime

# ==========================================
# 2025-2026 最新公式ドキュメント準拠検証結果
# Pandas: https://pandas.pydata.org/docs/
# ==========================================

pd.options.mode.chained_assignment = None

DATA_DIR: Final[str] = "Colog_github"

# --- パラメータ設定 ---
HOLDING_PERIOD: Final[int] = 5       # エントリー後の保有日数（例: 5営業日）
STOP_LOSS_PCT: Final[float] = 0.05   # ストップロスライン（ショートなので、エントリー価格から5%上昇で損切り）
TARGET_PROFIT: Final[float] = 0.10   # 利益確定ライン（エントリー価格から10%下落で利確）

def debug_log(msg: str) -> None:
    if not isinstance(msg, str): 
        raise TypeError("msg must be a string")
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] {msg}")

def load_and_merge_data(ticker: str) -> pd.DataFrame:
    """日足、信用残、財務データを読み込み、日付で結合して前日補完(ffill)する"""
    daily_file = f"{DATA_DIR}/{ticker}.parquet"
    margin_file = f"{DATA_DIR}/{ticker}_margin.parquet"
    fins_file = f"{DATA_DIR}/{ticker}_fins.parquet"
    
    if not os.path.exists(daily_file):
        return pd.DataFrame()
        
    df_daily = pd.read_parquet(daily_file)
    if df_daily.empty or 'date' not in df_daily.columns:
        return pd.DataFrame()
        
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.sort_values('date').reset_index(drop=True)
    
    # 信用残データの結合 (列の存在チェックによる安全なアクセス)
    if os.path.exists(margin_file):
        df_margin = pd.read_parquet(margin_file)
        if not df_margin.empty and 'date' in df_margin.columns:
            df_margin['date'] = pd.to_datetime(df_margin['date'])
            
            # 必須列が存在するか物理的に確認
            if 'LongMarginRatio' in df_margin.columns and 'ShortMarginRatio' in df_margin.columns:
                df_margin['mr_ratio'] = pd.to_numeric(df_margin['LongMarginRatio'], errors='coerce') / pd.to_numeric(df_margin['ShortMarginRatio'], errors='coerce').replace(0, np.nan)
                df_margin = df_margin[['date', 'mr_ratio']].dropna()
                df_daily = pd.merge(df_daily, df_margin, on='date', how='left')
    
    if 'mr_ratio' not in df_daily.columns:
        df_daily['mr_ratio'] = np.nan
    df_daily['mr_ratio'] = df_daily['mr_ratio'].ffill()
    
    # 財務データの結合 (列の存在チェックによる安全なアクセス)
    if os.path.exists(fins_file):
        df_fins = pd.read_parquet(fins_file)
        if not df_fins.empty and 'date' in df_fins.columns:
            df_fins['date'] = pd.to_datetime(df_fins['date'])
            df_fins = df_fins.sort_values('date')
            
            # NetSales列が存在しない場合はNaNのシリーズを生成
            net_sales_series = df_fins['NetSales'] if 'NetSales' in df_fins.columns else pd.Series(np.nan, index=df_fins.index)
            df_fins['NetSales_numeric'] = pd.to_numeric(net_sales_series, errors='coerce')
            df_fins['rev_growth'] = df_fins['NetSales_numeric'].pct_change() * 100
            
            if 'EPS' not in df_fins.columns:
                df_fins['EPS'] = np.nan
            if 'BPS' not in df_fins.columns:
                df_fins['BPS'] = np.nan
                
            df_fins = df_fins[['date', 'EPS', 'BPS', 'rev_growth']].dropna(how='all')
            df_daily = pd.merge(df_daily, df_fins, on='date', how='left')
            
    for col in ['EPS', 'BPS', 'rev_growth']:
        if col not in df_daily.columns:
            df_daily[col] = np.nan
        df_daily[col] = df_daily[col].ffill()
        
    return df_daily

def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """バックテストに必要なテクニカル・ファンダメンタル指標を一括計算"""
    # 200日未満のデータしかない銘柄は長期指標が計算できないためスキップ
    if df.empty or len(df) < 200:
        return df
        
    ps = df['close']
    
    # 1. RSI (14日) と スロープ
    dt = ps.diff()
    gain = dt.where(dt > 0, 0.0).rolling(window=14).mean()
    loss = -dt.where(dt < 0, 0.0).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
    df['rsi_slope'] = df['rsi'].diff(5)
    
    # 2. ボリンジャーバンド & 移動平均乖離率
    m20 = ps.rolling(20).mean()
    std20 = ps.rolling(20).std()
    df['bb_upper'] = m20 + (std20 * 2)
    
    m25 = ps.rolling(25).mean()
    df['d25'] = (ps - m25) / m25 * 100
    
    # 3. MACD
    ema12 = ps.ewm(span=12, adjust=False).mean()
    ema26 = ps.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - macd_signal
    df['macd_hist_prev'] = df['macd_hist'].shift(1)
    
    # 4. 信用倍率のZスコア (過去200日基準)
    if 'mr_ratio' in df.columns:
        mean_mr = df['mr_ratio'].rolling(200).mean()
        std_mr = df['mr_ratio'].rolling(200).std()
        df['mr_zscore'] = (df['mr_ratio'] - mean_mr) / std_mr.replace(0, np.nan)
        
    # 5. バリュエーション (PER / PBR)
    df['EPS_num'] = pd.to_numeric(df.get('EPS', np.nan), errors='coerce')
    df['BPS_num'] = pd.to_numeric(df.get('BPS', np.nan), errors='coerce')
    df['per'] = ps / df['EPS_num'].replace(0, np.nan)
    df['pbr'] = ps / df['BPS_num'].replace(0, np.nan)
    
    return df

def generate_short_signals(df: pd.DataFrame) -> pd.DataFrame:
    """ベクトル演算による高速なショートシグナル判定"""
    if df.empty or 'close' not in df.columns:
        df['short_signal'] = 0
        return df
        
    score = pd.Series(0, index=df.index, dtype=int)
    
    # 割高・業績悪化
    score += np.where(df.get('per', 0) > 40.0, 20, 0)
    score += np.where(df.get('pbr', 0) > 3.0, 15, 0)
    score += np.where(df.get('rev_growth', 0) < 0, 25, 0)
    
    # テクニカル過熱反落
    cond_rsi = (df.get('rsi', 0) > 60.0) & (df.get('rsi_slope', 0) < -5.0)
    score += np.where(cond_rsi, 30, 0)
    
    cond_bb = df['close'] > (df.get('bb_upper', np.inf) * 0.95)
    score += np.where(cond_bb, 15, 0)
    
    score += np.where(df.get('d25', 0) > 15.0, 15, 0)
    
    # MACDデッドクロス
    cond_macd_cross = (df.get('macd_hist_prev', 0) > 0) & (df.get('macd_hist', 0) <= 0)
    score += np.where(cond_macd_cross, 30, 0)
    cond_macd_down = (df.get('macd_hist_prev', 0) > df.get('macd_hist', 0)) & ~cond_macd_cross
    score += np.where(cond_macd_down, 10, 0)
    
    # 需給悪化
    score += np.where(df.get('mr_zscore', 0) >= 1.0, 20, 0)
    score += np.where(df.get('mr_ratio', 0) > 5.0, 20, 0)
    
    df['short_score'] = score
    df['short_signal'] = (score >= 50).astype(int)
    
    return df

def simulate_trades(df: pd.DataFrame, ticker: str) -> List[Dict[str, Any]]:
    """シグナル発生後の仮想トレードを実行し、結果を記録する"""
    trades = []
    if df.empty or 'short_signal' not in df.columns:
        return trades
        
    signal_indices = df.index[df['short_signal'] == 1].tolist()
    last_exit_idx = -1
    
    for idx in signal_indices:
        # ポジション保有中は追加エントリーしない（ピラミッディングなし）
        if idx <= last_exit_idx:
            continue
            
        entry_idx = idx + 1
        if entry_idx >= len(df):
            break
            
        entry_price = df['open'].iloc[entry_idx]
        entry_date = df['date'].iloc[entry_idx]
        
        if pd.isna(entry_price) or entry_price <= 0:
            continue
            
        # デフォルトのエグジット（時間経過）
        exit_idx = min(entry_idx + HOLDING_PERIOD - 1, len(df) - 1)
        exit_price = df['close'].iloc[exit_idx]
        exit_reason = "time_stop"
        
        # 期間中のストップロス（損切り）とテイクプロフィット（利確）の判定
        for j in range(entry_idx, exit_idx + 1):
            high_p = df['high'].iloc[j]
            low_p = df['low'].iloc[j]
            
            # ショートの場合、株価「上昇」が損失
            if high_p >= entry_price * (1 + STOP_LOSS_PCT):
                exit_idx = j
                exit_price = entry_price * (1 + STOP_LOSS_PCT)
                exit_reason = "stop_loss"
                break
                
            # ショートの場合、株価「下落」が利益
            if low_p <= entry_price * (1 - TARGET_PROFIT):
                exit_idx = j
                exit_price = entry_price * (1 - TARGET_PROFIT)
                exit_reason = "take_profit"
                break
                
        exit_date = df['date'].iloc[exit_idx]
        
        # ショートリターン: (売値 - 買値) / 売値
        ret = (entry_price - exit_price) / entry_price
        
        trades.append({
            'ticker': ticker,
            'entry_date': entry_date.strftime("%Y-%m-%d"),
            'exit_date': exit_date.strftime("%Y-%m-%d"),
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'return_pct': round(ret * 100, 2),
            'reason': exit_reason,
            'holding_days': exit_idx - entry_idx + 1
        })
        last_exit_idx = exit_idx
        
    return trades

def main() -> None:
    debug_log("🚀 ショート戦略バックテスト・シミュレーター起動")
    
    if not os.path.exists(DATA_DIR):
        debug_log(f"❌ データディレクトリ {DATA_DIR} が見つかりません。")
        return
        
    daily_files = glob.glob(f"{DATA_DIR}/????.parquet")
    all_trades: List[Dict[str, Any]] = []
    
    debug_log(f"📊 処理対象銘柄数: {len(daily_files)}銘柄")
    
    for i, file_path in enumerate(daily_files):
        ticker = os.path.basename(file_path).replace('.parquet', '')
        if not ticker.isdigit():
            continue
            
        df = load_and_merge_data(ticker)
        df = calculate_technical_features(df)
        df = generate_short_signals(df)
        trades = simulate_trades(df, ticker)
        
        if trades:
            all_trades.extend(trades)
            
        if (i + 1) % 50 == 0:
            debug_log(f"  - 進捗: {i + 1} / {len(daily_files)} 銘柄完了")
            
    if not all_trades:
        debug_log("⚠️ 条件に合致するトレードが1件も発生しませんでした。")
        return
        
    # --- パフォーマンスの集計 ---
    df_trades = pd.DataFrame(all_trades)
    
    total_trades = len(df_trades)
    winning_trades = df_trades[df_trades['return_pct'] > 0]
    losing_trades = df_trades[df_trades['return_pct'] <= 0]
    
    win_rate = (len(winning_trades) / total_trades) * 100
    avg_return = df_trades['return_pct'].mean()
    
    gross_profit = winning_trades['return_pct'].sum()
    gross_loss = abs(losing_trades['return_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    debug_log("=========================================")
    debug_log(f"📈 バックテスト結果サマリ (ショート戦略)")
    debug_log(f" - 総トレード数: {total_trades} 回")
    debug_log(f" - 勝率: {win_rate:.2f} %")
    debug_log(f" - 1トレード平均利益: {avg_return:.2f} %")
    debug_log(f" - プロフィットファクター (PF): {profit_factor:.2f}")
    debug_log("=========================================")
    
    df_trades.to_csv("backtest_report.csv", index=False)
    debug_log("✅ 全トレード履歴を backtest_report.csv に出力しました。")

# ==========================================
# 空データ・異常値に対する堅牢性証明テスト
# ==========================================
def run_integrity_tests() -> None:
    print("🧪 バリデーション中...")
    df_empty = pd.DataFrame()
    assert calculate_technical_features(df_empty).empty, "calc_tech failed on empty df"
    assert generate_short_signals(df_empty).empty, "generate_signals failed on empty df"
    assert len(simulate_trades(df_empty, "9999")) == 0, "simulate_trades failed on empty df"
    
    # 欠損カラムに対する耐性テスト (早期リターン条件である200行以上のダミーデータを生成して検証)
    df_missing_cols = pd.DataFrame({
        'date': pd.date_range(start='2025-01-01', periods=205),
        'close': np.ones(205) * 1000.0
    })
    assert 'rsi' in calculate_technical_features(df_missing_cols).columns, "Should handle missing cols gracefully"
    
    print("✅ 全検証合格。")

if __name__ == "__main__":
    run_integrity_tests()
    main()
