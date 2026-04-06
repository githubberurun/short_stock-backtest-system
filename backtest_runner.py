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

# --- パラメータ設定 (クロスセクション統合版) ---
MAX_SHORTS_PER_DAY: Final[int] = 5   # 1日にショートを仕掛ける最大銘柄数（ワーストランキング上位）
HOLDING_PERIOD: Final[int] = 5       # エントリー後の保有日数
STOP_LOSS_PCT: Final[float] = 0.05   # ストップロス（逆行5%で損切り）
TARGET_PROFIT: Final[float] = 0.10   # 利益確定（順行10%で利確）

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
    
    if os.path.exists(margin_file):
        df_margin = pd.read_parquet(margin_file)
        if not df_margin.empty and 'date' in df_margin.columns:
            df_margin['date'] = pd.to_datetime(df_margin['date'])
            if 'LongMarginRatio' in df_margin.columns and 'ShortMarginRatio' in df_margin.columns:
                df_margin['mr_ratio'] = pd.to_numeric(df_margin['LongMarginRatio'], errors='coerce') / pd.to_numeric(df_margin['ShortMarginRatio'], errors='coerce').replace(0, np.nan)
                df_margin = df_margin[['date', 'mr_ratio']].dropna()
                df_daily = pd.merge(df_daily, df_margin, on='date', how='left')
    
    if 'mr_ratio' not in df_daily.columns:
        df_daily['mr_ratio'] = np.nan
    df_daily['mr_ratio'] = df_daily['mr_ratio'].ffill()
    
    if os.path.exists(fins_file):
        df_fins = pd.read_parquet(fins_file)
        if not df_fins.empty and 'date' in df_fins.columns:
            df_fins['date'] = pd.to_datetime(df_fins['date'])
            df_fins = df_fins.sort_values('date')
            
            net_sales_series = df_fins['NetSales'] if 'NetSales' in df_fins.columns else pd.Series(np.nan, index=df_fins.index)
            df_fins['NetSales_numeric'] = pd.to_numeric(net_sales_series, errors='coerce')
            df_fins['rev_growth'] = df_fins['NetSales_numeric'].pct_change() * 100
            
            if 'EPS' not in df_fins.columns: df_fins['EPS'] = np.nan
            if 'BPS' not in df_fins.columns: df_fins['BPS'] = np.nan
                
            df_fins = df_fins[['date', 'EPS', 'BPS', 'rev_growth']].dropna(how='all')
            df_daily = pd.merge(df_daily, df_fins, on='date', how='left')
            
    for col in ['EPS', 'BPS', 'rev_growth']:
        if col not in df_daily.columns:
            df_daily[col] = np.nan
        df_daily[col] = df_daily[col].ffill()
        
    return df_daily

def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """バックテストに必要な指標を一括計算（25日・75日移動平均を必須化）"""
    if df.empty or len(df) < 100:  # IPO直後を弾くための最低期間
        return df
        
    ps = df['close']
    
    dt = ps.diff()
    gain = dt.where(dt > 0, 0.0).rolling(window=14).mean()
    loss = -dt.where(dt < 0, 0.0).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
    
    # トレンド判定の要（移動平均線）
    df['m25'] = ps.rolling(25).mean()
    df['m75'] = ps.rolling(75).mean()
    df['d25'] = (ps - df['m25']) / df['m25'] * 100
    
    # MACD
    ema12 = ps.ewm(span=12, adjust=False).mean()
    ema26 = ps.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - macd_signal
    df['macd_hist_prev'] = df['macd_hist'].shift(1)
    
    if 'mr_ratio' in df.columns:
        df['mr_zscore'] = (df['mr_ratio'] - df['mr_ratio'].rolling(200).mean()) / df['mr_ratio'].rolling(200).std().replace(0, np.nan)
        
    df['EPS_num'] = pd.to_numeric(df.get('EPS', np.nan), errors='coerce')
    df['BPS_num'] = pd.to_numeric(df.get('BPS', np.nan), errors='coerce')
    df['per'] = ps / df['EPS_num'].replace(0, np.nan)
    df['pbr'] = ps / df['BPS_num'].replace(0, np.nan)
    
    return df

def generate_scoring_and_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """厳格なダウントレンド判定と、ショート脆弱性スコアの算出"""
    if df.empty or 'close' not in df.columns or 'm75' not in df.columns:
        df['is_eligible'] = False
        df['short_score'] = 0
        return df
        
    score = pd.Series(0, index=df.index, dtype=int)
    
    # ファンダメンタルズの悪化
    score += np.where(df.get('per', 0) > 40.0, 20, 0)
    score += np.where(df.get('pbr', 0) > 3.0, 10, 0)
    score += np.where(df.get('rev_growth', 1) < 0, 30, 0) # 成長鈍化はショートの最大の味方
    
    # 需給の悪化（買い残の滞留）
    score += np.where(df.get('mr_zscore', 0) > 1.0, 20, 0) 
    
    # モメンタムの崩壊（MACDデッドクロス）
    cond_macd_cross = (df.get('macd_hist_prev', 0) > 0) & (df.get('macd_hist', 0) <= 0)
    score += np.where(cond_macd_cross, 20, 0)
    
    df['short_score'] = score
    
    # 【最重要】鉄壁のダウントレンド・フィルター
    # 株価が75日線の下、かつ25日線が75日線の下にある（上昇相場を完全に否定）
    is_downtrend = (df['close'] < df['m75']) & (df['m25'] < df['m75']) & (df['close'] < df['m25'])
    
    # フィルターを通過し、かつ一定のスコアを持つ日だけを「候補」とする
    df['is_eligible'] = is_downtrend & (score >= 30)
    
    return df

def main() -> None:
    debug_log("🚀 クロスセクション・ショート戦略シミュレーター起動")
    
    if not os.path.exists(DATA_DIR):
        debug_log(f"❌ データディレクトリ {DATA_DIR} が見つかりません。")
        return
        
    daily_files = glob.glob(f"{DATA_DIR}/????.parquet")
    
    # 全銘柄のデータをメモリに保持する辞書と、ランキング用のメタデータリスト
    data_dict: Dict[str, pd.DataFrame] = {}
    all_scores_list: List[pd.DataFrame] = []
    
    debug_log(f"📊 データロード＆スコアリング開始 (対象: {len(daily_files)}銘柄)")
    
    for i, file_path in enumerate(daily_files):
        ticker = os.path.basename(file_path).replace('.parquet', '')
        if not ticker.isdigit(): continue
            
        df = load_and_merge_data(ticker)
        df = calculate_technical_features(df)
        df = generate_scoring_and_eligibility(df)
        
        if not df.empty:
            data_dict[ticker] = df
            # ランキング計算用に必要な列だけを抽出してリストに追加（メモリ節約）
            df_meta = df[['date', 'short_score', 'is_eligible']].copy()
            df_meta['ticker'] = ticker
            df_meta['original_idx'] = df_meta.index # トレード実行時のインデックス検索を高速化
            all_scores_list.append(df_meta)
            
        if (i + 1) % 100 == 0:
            debug_log(f"  - 進捗: {i + 1} / {len(daily_files)} 銘柄完了")
            
    if not all_scores_list:
        debug_log("⚠️ 分析可能なデータがありません。")
        return
        
    debug_log("🏆 クロスセクション・ランキングを生成中...")
    df_all_scores = pd.concat(all_scores_list, ignore_index=True)
    
    # 候補に挙がった日のみを抽出
    df_candidates = df_all_scores[df_all_scores['is_eligible']].copy()
    
    # 日付ごとにスコアの降順でランキング付け（同点の場合は元のインデックス順等）
    df_candidates['rank'] = df_candidates.groupby('date')['short_score'].rank(method='first', ascending=False)
    
    # 各日において、スコアワースト上位 MAX_SHORTS_PER_DAY 銘柄のみを真のシグナルとする
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
        
        # 期間中のストップロスとテイクプロフィットの判定
        for j in range(entry_idx, exit_idx + 1):
            high_p = df_ticker.at[j, 'high']
            low_p = df_ticker.at[j, 'low']
            
            if high_p >= entry_price * (1 + STOP_LOSS_PCT):
                exit_idx = j
                exit_price = entry_price * (1 + STOP_LOSS_PCT)
                exit_reason = "stop_loss"
                break
                
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
    debug_log(f"📈 バックテスト結果サマリ (クロスセクション・ショート戦略)")
    debug_log(f" - 総トレード数: {total_trades} 回 (激減しているはずです)")
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
    
    # クロスセクション判定用のダミーデータテスト
    df_dummy = pd.DataFrame({
        'date': pd.date_range(start='2025-01-01', periods=150),
        'close': np.linspace(2000, 1000, 150) # 明確なダウントレンド
    })
    df_tech = calculate_technical_features(df_dummy)
    df_scored = generate_scoring_and_eligibility(df_tech)
    assert 'is_eligible' in df_scored.columns, "Should generate eligibility flag"
    
    print("✅ 全検証合格。")

if __name__ == "__main__":
    run_integrity_tests()
    main()
