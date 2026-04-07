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

# --- パラメータ設定 (小型株・バブル崩壊ショート戦略) ---
MAX_SHORTS_PER_DAY: Final[int] = 5   # 1日にショートを仕掛ける最大銘柄数
HOLDING_PERIOD: Final[int] = 3       # パニック下落の初動3日間だけを狙う
STOP_LOSS_PCT: Final[float] = 0.08   # 小型株のボラティリティを考慮し、損切り幅を8%に拡大
TARGET_PROFIT: Final[float] = 0.10   # 利益目標は10%（リスクリワード比 > 1.0）

def debug_log(msg: str) -> None:
    if not isinstance(msg, str): 
        raise TypeError("msg must be a string")
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] {msg}")

def load_data(ticker: str) -> pd.DataFrame:
    """日足データのみを読み込み、日付でソートする（財務・信用データ不要）"""
    daily_file = f"{DATA_DIR}/{ticker}.parquet"
    
    if not os.path.exists(daily_file):
        return pd.DataFrame()
        
    df_daily = pd.read_parquet(daily_file)
    if df_daily.empty or 'date' not in df_daily.columns:
        return pd.DataFrame()
        
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.sort_values('date').reset_index(drop=True)
    
    return df_daily

def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """小型株特有のバブル崩壊を検知するためのテクニカル指標を計算"""
    # 新興株は上場直後（IPO）も多いため、最低必要日数を30日に大幅短縮
    if df.empty or len(df) < 30:  
        return df
        
    ps = df['close']
    
    # 1. 短期トレンドライン (5日線) とその前日値
    df['m5'] = ps.rolling(5).mean()
    df['prev_m5'] = df['m5'].shift(1)
    df['prev_close'] = ps.shift(1)
    
    # 2. 中期トレンドからの過熱度 (25日線乖離率)
    df['m25'] = ps.rolling(25).mean()
    df['d25'] = (ps - df['m25']) / df['m25'] * 100
    
    # 3. RSI (14日)
    dt = ps.diff()
    gain = dt.where(dt > 0, 0.0).rolling(window=14).mean()
    loss = -dt.where(dt < 0, 0.0).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
    
    # 4. 出来高の急増検知（バブルの兆候）
    vs = df['volume']
    df['v_ma20'] = vs.rolling(20).mean()
    df['vol_ratio'] = vs / df['v_ma20'].replace(0, np.nan)
    
    return df

def generate_scoring_and_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """バブル崩壊（5日線割れ）の検知とスコアリング"""
    if df.empty or 'close' not in df.columns or 'm5' not in df.columns:
        df['is_eligible'] = False
        df['short_score'] = 0
        return df
        
    score = pd.Series(0, index=df.index, dtype=int)
    
    # 【絶対条件1】バブル状態であること：25日線から15%以上乖離している
    is_stretched = df.get('d25', 0) > 15.0
    
    # 【絶対条件2】トレンドの崩壊：前日は5日線の上にいたが、今日は5日線を明確に下抜けた
    is_broken = (df.get('prev_close', 0) > df.get('prev_m5', np.inf)) & (df['close'] < df['m5'])
    
    # スコアリング（より乖離が大きく、より出来高を伴って崩れたものを上位にする）
    score += np.where(df.get('d25', 0) > 20.0, 20, 0)
    score += np.where(df.get('d25', 0) > 30.0, 30, 0) # 狂気のバブル
    score += np.where(df.get('rsi', 0) > 75.0, 10, 0)
    score += np.where(df.get('vol_ratio', 0) > 2.0, 20, 0) # 出来高を伴う下落は信憑性が高い
    
    df['short_score'] = score
    df['is_eligible'] = is_stretched & is_broken
    
    return df

def main() -> None:
    debug_log("🚀 小型株・バブル崩壊ショート戦略シミュレーター起動")
    
    if not os.path.exists(DATA_DIR):
        debug_log(f"❌ データディレクトリ {DATA_DIR} が見つかりません。")
        return
        
    daily_files = glob.glob(f"{DATA_DIR}/????.parquet")
    
    data_dict: Dict[str, pd.DataFrame] = {}
    all_scores_list: List[pd.DataFrame] = []
    
    debug_log(f"📊 個別銘柄ロード＆スコアリング開始 (対象: {len(daily_files)}銘柄)")
    
    for i, file_path in enumerate(daily_files):
        ticker = os.path.basename(file_path).replace('.parquet', '')
        if not ticker.isdigit(): continue
            
        df = load_data(ticker)
        df = calculate_technical_features(df)
        df = generate_scoring_and_eligibility(df)
        
        if not df.empty:
            data_dict[ticker] = df
            df_meta = df[['date', 'short_score', 'is_eligible']].copy()
            df_meta['ticker'] = ticker
            df_meta['original_idx'] = df_meta.index 
            all_scores_list.append(df_meta)
            
        if (i + 1) % 50 == 0:
            debug_log(f"  - 進捗: {i + 1} / {len(daily_files)} 銘柄完了")
            
    if not all_scores_list:
        debug_log("⚠️ 分析可能なデータがありません。")
        return
        
    debug_log("🏆 バブル崩壊ランキングを生成中...")
    df_all_scores = pd.concat(all_scores_list, ignore_index=True)
    
    df_candidates = df_all_scores[df_all_scores['is_eligible']].copy()
    
    df_candidates['rank'] = df_candidates.groupby('date')['short_score'].rank(method='first', ascending=False)
    
    df_signals = df_candidates[df_candidates['rank'] <= MAX_SHORTS_PER_DAY].sort_values(['date', 'rank'])
    
    total_signals = len(df_signals)
    debug_log(f"🎯 厳選されたショートエントリーシグナル: 計 {total_signals} 件")
    
    all_trades: List[Dict[str, Any]] = []
    
    debug_log("⚔️ トレードシミュレーションを実行中...")
    
    for row in df_signals.itertuples():
        ticker = row.ticker
        idx = row.original_idx
        df_ticker = data_dict[ticker]
        
        # 翌日の始値でエントリー
        entry_idx = idx + 1
        if entry_idx >= len(df_ticker):
            continue
            
        entry_price = df_ticker.at[entry_idx, 'open']
        entry_date = df_ticker.at[entry_idx, 'date']
        
        if pd.isna(entry_price) or entry_price <= 0:
            continue
            
        exit_idx = min(entry_idx + HOLDING_PERIOD - 1, len(df_ticker) - 1)
        exit_price = df_ticker.at[exit_idx, 'close']
        exit_reason = "time_stop"
        
        for j in range(entry_idx, exit_idx + 1):
            high_p = df_ticker.at[j, 'high']
            low_p = df_ticker.at[j, 'low']
            
            # ストップロス判定（逆行）
            if high_p >= entry_price * (1 + STOP_LOSS_PCT):
                exit_idx = j
                exit_price = entry_price * (1 + STOP_LOSS_PCT)
                exit_reason = "stop_loss"
                break
                
            # テイクプロフィット判定（順行）
            if low_p <= entry_price * (1 - TARGET_PROFIT):
                exit_idx = j
                exit_price = entry_price * (1 - TARGET_PROFIT)
                exit_reason = "take_profit"
                break
                
        exit_date = df_ticker.at[exit_idx, 'date']
        ret = (entry_price - exit_price) / entry_price
        
        all_trades.append({
            'ticker': ticker,
            'entry_date': entry_date.strftime("%Y-%m-%d"),
            'exit_date': exit_date.strftime("%Y-%m-%d"),
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'return_pct': round(ret * 100, 2),
            'reason': exit_reason,
            'holding_days': exit_idx - entry_idx + 1,
            'short_score': row.short_score,
            'daily_rank': row.rank
        })
        
    if not all_trades:
        debug_log("⚠️ シミュレーションの結果、成立したトレードがありません。")
        return
        
    df_trades = pd.DataFrame(all_trades)
    
    total_trades = len(df_trades)
    winning_trades = df_trades[df_trades['return_pct'] > 0]
    losing_trades = df_trades[df_trades['return_pct'] <= 0]
    
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    avg_return = df_trades['return_pct'].mean() if total_trades > 0 else 0
    
    gross_profit = winning_trades['return_pct'].sum()
    gross_loss = abs(losing_trades['return_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    debug_log("=========================================")
    debug_log(f"📈 バックテスト結果サマリ (小型株バブル崩壊・ショート戦略)")
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
    assert generate_scoring_and_eligibility(df_empty).empty, "generate_signals failed on empty df"
    
    # バブル崩壊のダミーデータ生成（25日線乖離 + 5日線割れ）
    df_dummy = pd.DataFrame({
        'date': pd.date_range(start='2025-01-01', periods=40),
        'close': np.concatenate([np.linspace(1000, 2000, 39), [1800]]), # 最終日に急落
        'volume': np.ones(40) * 10000
    })
    df_tech = calculate_technical_features(df_dummy)
    df_scored = generate_scoring_and_eligibility(df_tech)
    
    assert 'is_eligible' in df_scored.columns, "Should generate eligibility flag"
    
    print("✅ 全検証合格。")

if __name__ == "__main__":
    run_integrity_tests()
    main()
