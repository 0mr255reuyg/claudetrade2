"""
BIST Swing Trader - Emre'nin Kişisel Borsa Uygulaması
Teknik + Temel Analiz ile En İyi 5 Hisseyi Puanlar
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# BIST 100 HİSSE LİSTESİ
# ─────────────────────────────────────────────
BIST100_TICKERS = [
    "ACSEL","ADEL","ADNAC","AKBNK","AKCNS","AKFGY","AKFYE","AKSA","AKSEN","AKSGY",
    "AKTAE","ALARK","ALBRK","ALFAS","ALGYO","ALKIM","ALKLC","ANELE","ANHYT","ARCLK",
    "ARDYZ","ASELS","ASGYO","ASTOR","ATAKP","ATATP","AYDEM","AYGAZ","BAGFS","BANVT",
    "BERA","BIENY","BIMAS","BIZIM","BJKAS","BKENT","BRISA","BRYAT","BSOKE","BTCIM",
    "BUCIM","CANTE","CCOLA","CEMTS","CIMSA","CLEBI","CWENE","DESA","DOHOL","DYOBY",
    "ECILC","EGEEN","EGERB","EKGYO","ENERU","ENJSA","ENKAI","EREGL","ESCOM","EUPWR",
    "EUREN","FENER","FLAP","FMIZP","FROTO","GARAN","GENIL","GESAN","GLYHO","GOLTS",
    "GUBRF","GWIND","HALKB","HATEK","HEKTS","HLGYO","HRKET","HTTBT","HUNER","ICBCT",
    "IHLGM","IHLAS","ISGSY","ISCTR","ISKUR","ISMEN","ISYAT","IZFAS","IZMDC","JANTS",
    "KAPLM","KAREL","KARSN","KATMR","KCAER","KCHOL","KENT","KLNMA","KMPUR","KNFRT",
    "KONYA","KORDS","KOZAA","KOZAL","KRDMD","KRGYO","KRONT","KSTUR","KTLEV","KUTPO",
    "LOGO","LKMNH","MAALT","MAVI","MEPET","MGROS","MIATK","MIPAZ","MPARK","NETAS",
    "NTHOL","NTTUR","NUGYO","NUHCM","ODAS","ONCSM","ORCAY","OTKAR","OYAKC","OYLUM",
    "OZGYO","OZKGY","PAPIL","PARSN","PCILT","PEKGY","PENGD","PETKM","PGSUS","PINSU",
    "PKENT","POLHO","PRKAB","PRKME","PTOFS","RAYSG","RODRG","ROYAL","RTALB","RYSAS",
    "SAHOL","SASA","SELEC","SELGD","SISE","SKBNK","SMART","SMRTG","SNPAM","SOKM",
    "SUMAS","SUNTK","SUPRS","TAVHL","TBMAN","TCELL","TGSAS","THYAO","TKFEN","TKNSA",
    "TOASO","TRGYO","TRILC","TSKB","TTKOM","TTRAK","TUKAS","TUPRS","TURSG","ULUFA",
    "ULUSE","UNCRD","UYUM","VAKBN","VAKFN","VERUS","VESBE","VESTL","VKGYO","VRGYO",
    "YKBNK","YATAS","YEOTK","YKSLN","YUNSA","ZOREN","ZRGYO"
]

# ─────────────────────────────────────────────
# VERİ ÇEKME FONKSİYONLARI
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fiyat verisi çeker."""
    try:
        t = ticker if ticker.endswith(".IS") else ticker + ".IS"
        df = yf.download(t, period=period, progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            return pd.DataFrame()
        # MultiIndex düzeltmesi
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fundamental_data(ticker: str) -> dict:
    """Temel analiz verisini çeker."""
    try:
        t = ticker if ticker.endswith(".IS") else ticker + ".IS"
        info = yf.Ticker(t).info
        return {
            "pb": info.get("priceToBook", None),
            "pe": info.get("trailingPE", None),
            "market_cap": info.get("marketCap", None),
            "sector": info.get("sector", "Bilinmiyor"),
            "name": info.get("longName", ticker)
        }
    except Exception:
        return {"pb": None, "pe": None, "market_cap": None, "sector": "Bilinmiyor", "name": ticker}

# ─────────────────────────────────────────────
# TEKNİK İNDİKATÖRLER
# ─────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """EMA, RSI, MACD, ATR hesaplar."""
    df = df.copy()
    close = df["Close"]

    # EMA
    df["EMA50"]  = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ATR
    high, low = df["High"], df["Low"]
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Hacim ortalaması
    df["Vol_MA5"] = df["Volume"].rolling(5).mean()

    return df

# ─────────────────────────────────────────────
# PUANLAMA SİSTEMİ (100 PUAN)
# ─────────────────────────────────────────────

def score_ticker(ticker: str) -> dict | None:
    """
    Teknik (60p) + Temel (40p) = 100p
    """
    df = get_price_data(ticker)
    if df.empty:
        return None

    df = calculate_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    score  = 0
    detail = {}

    # ── TEKNİK (60p) ──────────────────────────
    # 1. Golden Cross bölgesi: EMA50 > EMA200 (15p)
    ema_cross = bool(last["EMA50"] > last["EMA200"])
    pts = 15 if ema_cross else 0
    score += pts
    detail["EMA Golden Cross"] = (pts, 15, ema_cross)

    # 2. Fiyat her iki EMA'nın üzerinde (10p)
    price_above = bool(last["Close"] > last["EMA50"] and last["Close"] > last["EMA200"])
    pts = 10 if price_above else 0
    score += pts
    detail["Fiyat > EMA50/200"] = (pts, 10, price_above)

    # 3. RSI 40-60 bandı (10p)
    rsi_val = float(last["RSI"])
    rsi_band = 40 <= rsi_val <= 65
    pts = 10 if rsi_band else 0
    score += pts
    detail[f"RSI Bandı (şu an: {rsi_val:.1f})"] = (pts, 10, rsi_band)

    # 4. RSI yukarı yönlü 40'ı kesti (15p) — momentum başlangıcı
    rsi_cross = bool(float(prev["RSI"]) < 40 and rsi_val >= 40)
    # Eğer son 5 barda kestiyse de puan ver
    if not rsi_cross and len(df) >= 6:
        window = df.iloc[-6:-1]["RSI"].values
        for i in range(len(window)-1):
            if window[i] < 40 <= window[i+1]:
                rsi_cross = True
                break
    pts = 15 if rsi_cross else 0
    score += pts
    detail["RSI 40 Kesimi (Momentum)"] = (pts, 15, rsi_cross)

    # 5. MACD histogram pozitife döndü (10p)
    macd_turn = bool(float(last["MACD_Hist"]) > 0 and float(prev["MACD_Hist"]) <= 0)
    if not macd_turn:
        macd_turn = bool(float(last["MACD_Hist"]) > 0 and float(last["MACD"]) > float(last["MACD_Signal"]))
    pts = 10 if macd_turn else 0
    score += pts
    detail["MACD Pozitif Dönüş"] = (pts, 10, macd_turn)

    # ── TEMEL (40p) ───────────────────────────
    fund = get_fundamental_data(ticker)

    # 6. PD/DD < 1.5 (15p)
    pb   = fund["pb"]
    pb_ok = pb is not None and 0 < pb < 1.5
    pts  = 15 if pb_ok else 0
    score += pts
    pb_str = f"{pb:.2f}" if pb else "N/A"
    detail[f"PD/DD < 1.5 (şu an: {pb_str})"] = (pts, 15, pb_ok)

    # 7. F/K < 15 (15p)
    pe   = fund["pe"]
    pe_ok = pe is not None and 0 < pe < 15
    pts  = 15 if pe_ok else 0
    score += pts
    pe_str = f"{pe:.1f}" if pe else "N/A"
    detail[f"F/K < 15 (şu an: {pe_str})"] = (pts, 15, pe_ok)

    # 8. Piyasa değeri > 1 milyar TL (10p)
    mc   = fund["market_cap"]
    mc_ok = mc is not None and mc > 1_000_000_000
    pts  = 10 if mc_ok else 0
    score += pts
    detail[f"Piyasa Değeri > 1B TL"] = (pts, 10, mc_ok)

    return {
        "ticker":     ticker,
        "name":       fund["name"],
        "sector":     fund["sector"],
        "score":      score,
        "score_pct":  round(score, 1),
        "detail":     detail,
        "last_price": float(last["Close"]),
        "rsi":        round(rsi_val, 1),
        "ema50":      float(last["EMA50"]),
        "ema200":     float(last["EMA200"]),
        "macd_hist":  float(last["MACD_Hist"]),
        "pb":         pb,
        "pe":         pe,
        "df":         df
    }

# ─────────────────────────────────────────────
# STREAMLIT ARAYÜZÜ
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="BIST Swing Trader",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');
    
    :root {
        --bg: #0a0e1a;
        --card: #111827;
        --border: #1f2937;
        --accent: #00d4aa;
        --accent2: #f59e0b;
        --red: #ef4444;
        --text: #e5e7eb;
        --muted: #6b7280;
    }

    .stApp { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }
    
    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4aa, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem; }

    .score-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .score-card:hover { border-color: var(--accent); }

    .ticker-name {
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--accent);
    }
    .company-name { color: var(--muted); font-size: 0.8rem; }

    .score-badge {
        font-family: 'Space Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .score-high  { color: #00d4aa; }
    .score-mid   { color: #f59e0b; }
    .score-low   { color: #ef4444; }

    .progress-bar {
        background: var(--border);
        border-radius: 99px;
        height: 6px;
        margin-top: 4px;
    }
    .progress-fill {
        border-radius: 99px;
        height: 6px;
        background: linear-gradient(90deg, #00d4aa, #f59e0b);
    }

    .tag {
        display: inline-block;
        background: #1f2937;
        color: var(--muted);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin-right: 4px;
    }
    .tag-green { color: #00d4aa; background: #00d4aa15; }
    .tag-red   { color: #ef4444; background: #ef444415; }

    [data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--border); }
    .stButton>button {
        background: linear-gradient(135deg, #00d4aa22, #00d4aa44);
        border: 1px solid var(--accent);
        color: var(--accent);
        font-family: 'Space Mono', monospace;
        border-radius: 8px;
        width: 100%;
        padding: 0.6rem;
        font-weight: 700;
        letter-spacing: 1px;
        transition: all 0.2s;
    }
    .stButton>button:hover { background: var(--accent); color: #000; }

    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 0.5rem;
    }
    .metric-item { text-align: center; }
    .metric-val { font-family: 'Space Mono', monospace; font-size: 0.95rem; font-weight: 700; }
    .metric-lbl { color: var(--muted); font-size: 0.7rem; }

    .detail-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0;
        border-bottom: 1px solid var(--border);
        font-size: 0.82rem;
    }
    .check-yes { color: #00d4aa; }
    .check-no  { color: #ef4444; }

    .stSelectbox label, .stSlider label, .stMultiSelect label {
        color: var(--muted) !important; font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Başlık
    st.markdown('<div class="main-title">📈 BIST SWING TRADER</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Teknik + Temel Analiz · 100 Puanlık Skorlama · En İyi 5 Hisse</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Ayarlar")
        
        num_stocks = st.slider("Taranacak hisse sayısı", 20, len(BIST100_TICKERS), 50,
                               help="Fazla hisse = daha uzun süre")
        top_n = st.slider("Gösterilecek en iyi hisse", 3, 10, 5)

        st.markdown("---")
        st.markdown("### 🎯 Filtreler")
        min_score = st.slider("Min. skor (%)", 0, 80, 30)
        
        st.markdown("---")
        st.markdown("### 📊 Teknik Kriter Ağırlıkları")
        st.caption("Puanlar kodda ayarlıdır. Güncel kriter özeti:")
        st.markdown("""
        <div style='font-size:0.75rem; color:#6b7280; line-height:1.8'>
        • EMA Golden Cross: <b style='color:#00d4aa'>15p</b><br>
        • Fiyat > EMA50/200: <b style='color:#00d4aa'>10p</b><br>
        • RSI 40-65 Bandı: <b style='color:#00d4aa'>10p</b><br>
        • RSI 40 Momentum Kesimi: <b style='color:#f59e0b'>15p</b><br>
        • MACD Pozitif Dönüş: <b style='color:#00d4aa'>10p</b><br>
        • PD/DD < 1.5: <b style='color:#f59e0b'>15p</b><br>
        • F/K < 15: <b style='color:#f59e0b'>15p</b><br>
        • Piyasa Değeri > 1B: <b style='color:#00d4aa'>10p</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        scan_btn = st.button("🔍 TARA", use_container_width=True)

    # Ana içerik
    if not scan_btn:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='score-card' style='text-align:center; padding: 2rem'>
                <div style='font-size:2.5rem'>🔍</div>
                <div style='font-family: Space Mono, monospace; color:#00d4aa; margin-top:0.5rem'>Sol menüden</div>
                <div style='color:#6b7280; font-size:0.85rem'>tarama başlat</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='score-card' style='text-align:center; padding: 2rem'>
                <div style='font-size:2.5rem'>📊</div>
                <div style='font-family: Space Mono, monospace; color:#f59e0b; margin-top:0.5rem'>100 puanlık</div>
                <div style='color:#6b7280; font-size:0.85rem'>skorlama sistemi</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class='score-card' style='text-align:center; padding: 2rem'>
                <div style='font-size:2.5rem'>🎯</div>
                <div style='font-family: Space Mono, monospace; color:#00d4aa; margin-top:0.5rem'>En iyi 5 hisse</div>
                <div style='color:#6b7280; font-size:0.85rem'>swing için seçilir</div>
            </div>""", unsafe_allow_html=True)
        return

    # Tarama
    tickers_to_scan = BIST100_TICKERS[:num_stocks]
    
    progress_bar = st.progress(0)
    status_text  = st.empty()
    results      = []

    for i, ticker in enumerate(tickers_to_scan):
        status_text.markdown(f"<span style='color:#6b7280; font-size:0.8rem'>🔍 {ticker} taranıyor... ({i+1}/{len(tickers_to_scan)})</span>",
                             unsafe_allow_html=True)
        progress_bar.progress((i+1) / len(tickers_to_scan))
        
        result = score_ticker(ticker)
        if result and result["score_pct"] >= min_score:
            results.append(result)

    progress_bar.empty()
    status_text.empty()

    if not results:
        st.warning("Kriterlere uyan hisse bulunamadı. Min. skoru düşürün.")
        return

    results_sorted = sorted(results, key=lambda x: x["score_pct"], reverse=True)[:top_n]

    # Özet metrikler
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Taranan Hisse",     len(tickers_to_scan))
    m2.metric("Kritere Uyan",      len(results))
    m3.metric("En Yüksek Skor",    f"{results_sorted[0]['score_pct']}%")
    m4.metric("Ort. Skor",         f"{np.mean([r['score_pct'] for r in results_sorted]):.1f}%")
    st.markdown("---")

    # Üst 5 hisse kartları
    st.markdown(f"### 🏆 En İyi {top_n} Swing Adayı")
    
    for rank, res in enumerate(results_sorted):
        s = res["score_pct"]
        score_class = "score-high" if s >= 70 else ("score-mid" if s >= 50 else "score-low")
        medal = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"][rank]
        
        with st.expander(f"{medal} **{res['ticker']}** — {res['name'][:40]}  |  Skor: **{s}%**", expanded=(rank < 3)):
            col_left, col_right = st.columns([3,2])
            
            with col_left:
                st.markdown("**Kriter Detayları**")
                for crit, (earned, max_pts, ok) in res["detail"].items():
                    icon = "✅" if ok else "❌"
                    bar = f"<span style='color:#00d4aa'>{earned}/{max_pts}p</span>" if ok else f"<span style='color:#6b7280'>0/{max_pts}p</span>"
                    st.markdown(f"<div class='detail-row'><span>{icon} {crit}</span>{bar}</div>",
                               unsafe_allow_html=True)
                
                # Puan barı
                fill = int(s)
                st.markdown(f"""
                <div style='margin-top:1rem'>
                    <div style='display:flex; justify-content:space-between; font-size:0.8rem'>
                        <span style='color:#6b7280'>Toplam Skor</span>
                        <span class='{score_class}' style='font-family:Space Mono,monospace; font-weight:700'>{s}%</span>
                    </div>
                    <div class='progress-bar'>
                        <div class='progress-fill' style='width:{fill}%'></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            
            with col_right:
                st.markdown("**Anlık Değerler**")
                vals = {
                    "Fiyat (TL)": f"{res['last_price']:.2f}",
                    "RSI":        f"{res['rsi']}",
                    "EMA50":      f"{res['ema50']:.2f}",
                    "EMA200":     f"{res['ema200']:.2f}",
                    "PD/DD":      f"{res['pb']:.2f}" if res['pb'] else "N/A",
                    "F/K":        f"{res['pe']:.1f}"  if res['pe'] else "N/A",
                }
                for k, v in vals.items():
                    st.markdown(f"<div class='detail-row'><span style='color:#6b7280'>{k}</span><span style='font-family:Space Mono,monospace'>{v}</span></div>",
                               unsafe_allow_html=True)
                
                # Mini fiyat grafiği
                df_plot = res["df"].tail(60).copy()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot["Close"],
                    line=dict(color="#00d4aa", width=1.5), name="Fiyat", showlegend=False))
                fig.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot["EMA50"],
                    line=dict(color="#f59e0b", width=1, dash="dot"), name="EMA50", showlegend=False))
                fig.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot["EMA200"],
                    line=dict(color="#ef4444", width=1, dash="dot"), name="EMA200", showlegend=False))
                fig.update_layout(
                    height=180, margin=dict(l=0,r=0,t=0,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False, tickfont=dict(size=8, color="#6b7280"))
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Karşılaştırma tablosu
    st.markdown("---")
    st.markdown("### 📋 Özet Tablo")
    
    table_data = []
    for res in results_sorted:
        table_data.append({
            "Hisse": res["ticker"],
            "Şirket": res["name"][:30],
            "Skor (%)": res["score_pct"],
            "Fiyat (TL)": round(res["last_price"], 2),
            "RSI": res["rsi"],
            "EMA50": round(res["ema50"], 2),
            "EMA200": round(res["ema200"], 2),
            "PD/DD": round(res["pb"], 2) if res["pb"] else "N/A",
            "F/K": round(res["pe"], 1) if res["pe"] else "N/A",
        })
    
    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)
    
    # Skor dağılım grafiği
    st.markdown("---")
    st.markdown("### 📊 Skor Dağılımı")
    fig2 = px.bar(
        df_table, x="Hisse", y="Skor (%)",
        color="Skor (%)",
        color_continuous_scale=["#ef4444","#f59e0b","#00d4aa"],
        range_color=[0,100]
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb", height=300,
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, range=[0,100]),
        coloraxis_showscale=False,
        margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div style='text-align:center; color:#374151; font-size:0.75rem; margin-top:2rem; font-family: Space Mono, monospace;'>
    ⚠️ Bu uygulama yatırım tavsiyesi değildir. Karar vermek sana aittir.
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
