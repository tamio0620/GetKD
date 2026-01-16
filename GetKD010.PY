import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import urllib3
import io
from datetime import datetime, timedelta

# 基本設定：關閉 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股類股 KD 門檻掃描器", layout="wide")

# --- 核心邏輯函數 ---

@st.cache_data(ttl=3600)
def get_category_data():
    """從證交所抓取上市股票清單與類別"""
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    try:
        res = requests.get(url, verify=False, timeout=15)
        html_data = io.StringIO(res.text)
        df = pd.read_html(html_data)[0]
        df.columns = df.iloc[0]
        df = df.iloc[2:]
        
        # 解析代碼與名稱
        df['代碼'] = df['有價證券代號及名稱'].str.split('　').str[0]
        df['名稱'] = df['有價證券代號及名稱'].str.split('　').str[1]
        
        # 取得所有產業類別
        categories = sorted(df['產業別'].dropna().unique().tolist())
        categories = [c for c in categories if "業" in c or "其" in c]
        return categories, df
    except Exception as e:
        st.error(f"無法獲取類股清單: {e}")
        return [], None

def calculate_indicators(df):
    """手動計算所有要求的技術指標"""
    # 處理 yfinance 可能產生的多層索引
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)

    # 1. KD (9, 3, 3) - 採用台股標準指數平滑法
    low_9 = low.rolling(window=9).min()
    high_9 = high.rolling(window=9).max()
    rsv = (close - low_9) / (high_9 - low_9) * 100
    
    k_vals, d_vals = [], []
    curr_k, curr_d = 50.0, 50.0
    for val in rsv:
        if pd.isna(val):
            k_vals.append(50.0); d_vals.append(50.0)
        else:
            curr_k = (curr_k * 2/3) + (val * 1/3)
            curr_d = (curr_d * 2/3) + (curr_k * 1/3)
            k_vals.append(curr_k); d_vals.append(curr_d)
    
    df = df.copy()
    df['K9'], df['D9'] = k_vals, d_vals

    # 2. MA6 / MA12
    df['MA6'] = close.rolling(window=6).mean()
    df['MA12'] = close.rolling(window=12).mean()

    # 3. BIAS6 (乖離率)
    df['BIAS6'] = ((close - df['MA6']) / df['MA6']) * 100

    # 4. RSI6
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    df['RSI6'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))

    # 5. WR12 (威廉指標)
    df['WR12'] = (high.rolling(window=12).max() - close) / (high.rolling(window=12).max() - low.rolling(window=12).min() + 1e-10) * -100

    # 6. MTM6 & MTM6_MA
    df['MTM6'] = close - close.shift(6)
    df['MTM6_MA'] = df['MTM6'].rolling(window=6).mean()

    return df

# --- 網頁介面佈局 ---

# 使用兩欄佈局處理標題與單位
title_col, unit_col = st.columns([4, 1])

with title_col:
    st.title("📈 台股類股技術指標掃描器")
    st.caption("⚠️ 投資有風險 請謹慎理財")

with unit_col:
    # 單位標註，設定 padding 讓它垂直對齊標題底部，字體設為 1.1rem (約 H1 的一半)
    st.markdown(
        """
        <div style="text-align: right; padding-top: 38px;">
            <span style="font-size: 1.1rem; color: #888888; font-weight: 500;">
                單位：工具機運動控制實驗室
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# 1. 初始化資料
categories, full_df = get_category_data()

# 2. 側邊欄設定
st.sidebar.header("📊 掃描參數設定")
if categories:
    selected_cat = st.sidebar.selectbox("1. 選擇產業類別", categories, 
                                        index=categories.index("半導體業") if "半導體業" in categories else 0)
    
    k_threshold_mode = st.sidebar.radio("2. K9 數值門檻限制", ["不設限", "自定義門檻"])
    k_threshold_val = 100.0
    if k_threshold_mode == "自定義門檻":
        k_threshold_val = st.sidebar.slider("設定 K9 必須小於等於", 10, 80, 20)
        st.sidebar.caption(f"💡 僅篩選 K9 ≦ {k_threshold_val} 的黃金交叉")

    lookback_days = st.sidebar.slider("3. 分析回溯天數", 60, 180, 100)
    start_btn = st.sidebar.button("🚀 開始執行掃描")

    # 3. 執行邏輯
    if start_btn:
        target_stocks = full_df[(full_df['產業別'] == selected_cat) & (full_df['代碼'].str.len() == 4)]
        stock_list = target_stocks[['代碼', '名稱']].to_dict('records')
        
        if not stock_list:
            st.warning(f"在 {selected_cat} 類別下找不到符合的上市股票。")
        else:
            st.info(f"正在掃描 {selected_cat} 類股 (共 {len(stock_list)} 檔)...")
            progress_bar = st.progress(0)
            results = []
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            for i, stock in enumerate(stock_list):
                code = stock['代碼']
                try:
                    df = yf.download(f"{code}.TW", start=start_date, progress=False, timeout=10)
                    if not df.empty and len(df) >= 20:
                        df = calculate_indicators(df)
                        curr, prev = df.iloc[-1], df.iloc[-2]
                        
                        # 判定條件：KD金叉 且 符合 K值門檻
                        is_gold_cross = prev['K9'] <= prev['D9'] and curr['K9'] > curr['D9']
                        is_under_threshold = curr['K9'] <= k_threshold_val
                        
                        if is_gold_cross and is_under_threshold:
                            results.append({
                                '股票代碼': code, '股票名稱': stock['名稱'],
                                '收盤價': round(float(curr['Close']), 2),
                                'K9': round(float(curr['K9']), 2), 'D9': round(float(curr['D9']), 2),
                                'MA6': round(float(curr['MA6']), 2), 'MA12': round(float(curr['MA12']), 2),
                                'BIAS6': round(float(curr['BIAS6']), 2), 'RSI6': round(float(curr['RSI6']), 2),
                                'WR12': round(float(curr['WR12']), 2), 'MTM6': round(float(curr['MTM6']), 2),
                                'MTM6_MA': round(float(curr['MTM6_MA']), 2)
                            })
                except:
                    pass
                progress_bar.progress((i + 1) / len(stock_list))
            
            # 4. 結果顯示
            st.subheader(f"🔍 {selected_cat} 篩選結果")
            if results:
                res_df = pd.DataFrame(results)
                st.success(f"成功！共有 {len(results)} 檔符合條件。")
                st.dataframe(res_df, use_container_width=True)
                
                # CSV 下載按鈕
                csv_data = res_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    label="📥 下載分析報表 (CSV)",
                    data=csv_data,
                    file_name=f"{selected_cat}_KD_Scan_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("查無符合條件之標的，請放寬門檻或更換類別。")
else:
    st.write("👈 請在左側設定條件後點擊開始掃描。")