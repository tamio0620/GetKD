# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# 網頁配置
st.set_page_config(page_title="技術指標分析儀 Pro", layout="wide")

st.title("N日技術指標自動化分析模組")
st.markdown("---")

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("輸入設定")
    ticker = st.text_input("輸入代碼 (例: 2330.TW)", value="2330.TW")
    target_date = st.date_input("基準日期", datetime.now())
    n_days = st.slider("前後觀察天數 (N)", min_value=5, max_value=60, value=20)
    fetch_button = st.button("生成分析圖表", type="primary")

# --- 數據計算函數 ---
def get_stock_data(ticker, target_date, n_days):
    start_date = target_date - timedelta(days=n_days + 100)
    end_date = target_date + timedelta(days=n_days + 10)
    
    try:
        raw_df = yf.download(ticker, start=start_date, end=end_date)
        if raw_df.empty: return None
        
        df = raw_df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        close = df['Close'].squeeze().astype(float)
        high = df['High'].squeeze().astype(float)
        low = df['Low'].squeeze().astype(float)
        open_price = df['Open'].squeeze().astype(float)

        # 指標計算
        df['MA6'] = close.rolling(window=6).mean()
        df['MA12'] = close.rolling(window=12).mean()
        df['BIAS6'] = ((close - df['MA6']) / df['MA6']) * 100
        ma3 = close.rolling(window=3).mean()
        df['BIAS_3_6_Diff'] = (((close - ma3) / ma3) * 100) - df['BIAS6']
        
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        df['RSI6'] = (up.rolling(6).mean() / (up.rolling(6).mean() + down.rolling(6).mean())) * 100

        low_min, high_max = low.rolling(9).min(), high.rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min) * 100
        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()

        df['WR12'] = (high.rolling(12).max() - close) / (high.rolling(12).max() - low.rolling(12).min()) * -100
        df['MTM6'] = close - close.shift(6)
        df['MTM6_MA'] = df['MTM6'].rolling(window=6).mean()

        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['DIF'] = exp1 - exp2
        df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()

        mask = (df.index >= pd.Timestamp(target_date - timedelta(days=n_days))) & \
               (df.index <= pd.Timestamp(target_date + timedelta(days=n_days)))
        return df.loc[mask]
    except Exception as e:
        st.error(f"數據處理錯誤: {str(e)}")
        return None

# --- 繪圖核心函數 ---
def create_chart(df, chart_type, title, note, chart_id):
    fig = go.Figure()
    
    if chart_type == "KD":
        fig.add_trace(go.Scatter(x=df.index, y=df['K'], name="K值"))
        fig.add_trace(go.Scatter(x=df.index, y=df['D'], name="D值"))
        up = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
        down = (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1))
        fig.add_trace(go.Scatter(x=df.index[up], y=df['K'][up], mode='markers+text', text="[買點]", textposition="top center", marker=dict(color='red', size=8)))
        fig.add_trace(go.Scatter(x=df.index[down], y=df['K'][down], mode='markers+text', text="[賣點]", textposition="bottom center", marker=dict(color='green', size=8)))

    elif chart_type == "MA":
        fig.add_trace(go.Scatter(x=df.index, y=df['MA6'], name="MA6"))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA12'], name="MA12"))
        up = (df['MA6'] > df['MA12']) & (df['MA6'].shift(1) <= df['MA12'].shift(1))
        down = (df['MA6'] < df['MA12']) & (df['MA6'].shift(1) >= df['MA12'].shift(1))
        fig.add_trace(go.Scatter(x=df.index[up], y=df['MA6'][up], mode='markers+text', text="[短期上漲動能增強]", textposition="top center", marker=dict(symbol='triangle-up', size=10)))
        fig.add_trace(go.Scatter(x=df.index[down], y=df['MA6'][down], mode='markers+text', text="[短期走弱下跌]", textposition="bottom center", marker=dict(symbol='triangle-down', size=10)))

    elif chart_type == "BIAS":
        fig.add_trace(go.Scatter(x=df.index, y=df['BIAS6'], name="BIAS6"))
        over_up = df['BIAS6'] > 3.5
        over_down = df['BIAS6'] < -3
        fig.add_trace(go.Scatter(x=df.index[over_up], y=df['BIAS6'][over_up], mode='markers+text', text="[超買]", textposition="top center", marker=dict(color='red')))
        fig.add_trace(go.Scatter(x=df.index[over_down], y=df['BIAS6'][over_down], mode='markers+text', text="[超賣]", textposition="bottom center", marker=dict(color='green')))

    elif chart_type == "DIFF":
        fig.add_trace(go.Scatter(x=df.index, y=df['BIAS_3_6_Diff'], name="Diff"))
        fig.add_hline(y=0, line_dash="dash")
        up = (df['BIAS_3_6_Diff'] > 0) & (df['BIAS_3_6_Diff'].shift(1) <= 0)
        down = (df['BIAS_3_6_Diff'] < 0) & (df['BIAS_3_6_Diff'].shift(1) >= 0)
        fig.add_trace(go.Scatter(x=df.index[up], y=[0]*sum(up), mode='markers+text', text="[即將回檔]", textposition="top center"))
        fig.add_trace(go.Scatter(x=df.index[down], y=[0]*sum(down), mode='markers+text', text="[即將反彈]", textposition="bottom center"))

    elif chart_type == "RSI":
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI6'], name="RSI6"))
        fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, annotation_text="[超買]區")
        fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, annotation_text="[超賣]區")

    elif chart_type == "WR":
        fig.add_trace(go.Scatter(x=df.index, y=df['WR12'], name="WR12"))
        fig.add_hrect(y0=-20, y1=0, fillcolor="red", opacity=0.1, annotation_text="[超買區-近高點]")
        fig.add_hrect(y0=-80, y1=-20, fillcolor="gray", opacity=0.1, annotation_text="[中性區]")
        fig.add_hrect(y0=-100, y1=-80, fillcolor="green", opacity=0.1, annotation_text="[超賣區-近低點]")

    elif chart_type == "MTM":
        # (1) 修正背景顏色參數
        for i in range(1, len(df)):
            color = "rgba(255,0,0,0.1)" if df['MTM6'].iloc[i] > 0 else "rgba(0,255,0,0.1)"
            fig.add_vrect(x0=df.index[i-1], x1=df.index[i], fillcolor=color, line_width=0)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['MTM6'], name="MTM6"))
        fig.add_trace(go.Scatter(x=df.index, y=df['MTM6_MA'], name="MTM6_MA", line=dict(dash='dot')))
        
        # (2) 突破0軸標記
        up0 = (df['MTM6'] > 0) & (df['MTM6'].shift(1) <= 0)
        down0 = (df['MTM6'] < 0) & (df['MTM6'].shift(1) >= 0)
        fig.add_trace(go.Scatter(x=df.index[up0], y=df['MTM6'][up0], mode='markers+text', text="[買進]", textposition="top center"))
        fig.add_trace(go.Scatter(x=df.index[down0], y=df['MTM6'][down0], mode='markers+text', text="[賣出]", textposition="bottom center"))
        
        # (3) 交叉點滑鼠顯示 (動能轉強/減弱)
        strong = (df['MTM6'] > df['MTM6_MA']) & (df['MTM6'].shift(1) <= df['MTM6_MA'].shift(1))
        weak = (df['MTM6'] < df['MTM6_MA']) & (df['MTM6'].shift(1) >= df['MTM6_MA'].shift(1))
        fig.add_trace(go.Scatter(x=df.index[strong], y=df['MTM6'][strong], mode='markers', hovertext="動能轉強(反彈或漲勢)-買進訊號", name="轉強"))
        fig.add_trace(go.Scatter(x=df.index[weak], y=df['MTM6'][weak], mode='markers', hovertext="動能減弱(回檔)-賣出訊號", name="減弱"))

    elif chart_type == "MACD":
        fig.add_trace(go.Scatter(x=df.index, y=df['DIF'], name="DIF"))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="DEA"))
        up = (df['DIF'] > df['MACD_Signal']) & (df['DIF'].shift(1) <= df['MACD_Signal'].shift(1))
        down = (df['DIF'] < df['MACD_Signal']) & (df['DIF'].shift(1) >= df['MACD_Signal'].shift(1))
        fig.add_trace(go.Scatter(x=df.index[up], y=df['DIF'][up], mode='markers+text', text="[多頭增強(買入訊號)]", textposition="top center"))
        fig.add_trace(go.Scatter(x=df.index[down], y=df['DIF'][down], mode='markers+text', text="[空頭增強(賣出訊號)]", textposition="bottom center"))

    elif chart_type == "VOL":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="收盤價"), secondary_y=False)
        colors = ['red' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'green' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="成交量", marker_color=colors, opacity=0.3), secondary_y=True)

    fig.update_layout(title=title, height=400, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.text_area("註解說明：", value=note, key=f"note_{chart_id}", height=70)

# --- 主程式 ---
if fetch_button:
    with st.spinner('分析中...'):
        data = get_stock_data(ticker, target_date, n_days)
        if data is not None:
            c_config = [
                #("KD", "圖 1: KD 買賣點標記", "K值由下往上穿越D值買點；由上往下穿越D值賣點", "c1"),
                ("KD", "圖 1: K9/D9 買賣點標記", "K值由下往上穿越D值是買入信號預示上漲；K值由上往下穿越D值是賣出信號預示下跌. 注意!確認賣出條件：D9(i)-D9(i-1)>30 and 股價diff為正", "c1"),
                #("MA", "圖 2: MA 趨勢標記", "MA6/MA12黃金交叉與死亡交叉標記", "c2"),
                ("MA", "圖 2: MA6/MA12 短期動能指標", "MA6上穿MA12是短期上漲動能增強；MA6下穿MA12是短期走弱預示下跌", "c2"),
                #("BIAS", "圖 3: BIAS6 超買超賣", "乖離率 >3.5% 超買；<-3% 超賣", "c3"),
                ("BIAS", "圖 3: BIAS6 6日乖離率 超買超賣指標", "乖離率在 3.5% 以上為超買;-3% 以下為超賣", "c3"),
                #("DIFF", "圖 4: BIAS Diff 轉折", "由負轉正標註[即將回檔]；由正轉負標註[即將反彈]", "c4"),
                ("DIFF", "圖 4: BIAS Diff 轉折指標", "當Diff值由負轉正且突破0軸時標註[即將回檔];當Diff值由正轉負且跌破0軸時標註[即將反彈]", "c4"),
                #("RSI", "圖 5: RSI 強弱區域", "70/30分界標註超買超賣區域", "c5"),
                ("RSI", "圖 5: RSI6 強弱指標", "RSI6大於70區域標註[超買]區;RSI6小於30區域標註[超賣]區", "c5"),
                #("WR", "圖 6: WR 威廉區域", "0~20超買；80~100超賣分區顯示", "c6"),
                ("WR", "圖 6: WR12 威廉指標", "0~20 超買區=>近高點=>即將回檔=>賣出 ;20~80 中性區 ;80~100 超賣區=>近低點=>即將反彈=>買進 ", "c6"),
                #("MTM", "圖 7: MTM 動能背景", "紅綠背景標記動能；滑鼠顯示強弱訊號", "c7"),
                ("MTM", "圖 7: MTM6 動能指標", "(1)MTM6>0是上升動能;MTM6<0是下跌動能. (2)MTM6由下往上突破0為[買進]訊號;MTM6由上向下跌破0軸時為[賣出]訊號. (3)MTM6由下往上突破MA6均線且皆位0軸下方時動能轉強(反彈或漲勢)是買進訊號;MTM6由上往下穿過MA6均線時動能減弱(回檔)是賣出訊號", "c7"),
                #("MACD", "圖 8: MACD 訊號", "DIF/DEA交叉點位標註多空訊號", "c8")
                ("MACD", "圖 8: MACD 多空訊號", "買入訊號=>DIF線（快線）向上突破DEA線（慢線）=>多頭增強;賣出訊號=>DIF線（快線）向下跌破DEA線（慢線）=>空頭增強", "c8")
            ]
            for i in range(0, 8, 2):
                cols = st.columns(2)
                for j in range(2):
                    with cols[j]:
                        create_chart(data, c_config[i+j][0], c_config[i+j][1], c_config[i+j][2], c_config[i+j][3])
            
            st.subheader("圖表 9: 收盤價與成交量分析")
            create_chart(data, "VOL", "收盤價 (曲線) 與 成交量 (量條)", "量增價漲顯示紅柱；量增價跌顯示綠柱", "c9")
