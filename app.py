import os
import json
import logging
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import google.generativeai as genai
import re
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logging.info("Gemini API configured successfully.")
    except Exception as e:
        logging.error(f"Failed to configure Gemini API: {e}")
else:
    logging.warning("CRITICAL: GEMINI_API_KEY not found. AI features will be disabled.")

app = Flask(__name__)

def convert_numpy_types(obj):
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if pd.isna(obj): return None
    return obj

def get_stock_symbols():
    try:
        df = pd.read_csv('Symbols.csv')
        symbols = df.iloc[:, 0].dropna().str.strip().unique().tolist()
        symbols_to_fetch = [s for s in symbols if s and len(s) < 15][:75]
        sectors = {}
        for symbol in symbols_to_fetch:
            try:
                ticker_info = yf.Ticker(f"{symbol}.NS").info
                sector = ticker_info.get('sector', 'Other')
                if sector not in sectors: sectors[sector] = []
                sectors[sector].append(symbol)
            except Exception:
                if 'Other' not in sectors: sectors['Other'] = []
                sectors['Other'].append(symbol)
        return {k: sorted(v) for k, v in sorted(sectors.items())}
    except Exception as e:
        logging.error(f"Error reading Symbols.csv: {e}")
        return {"Error": ["Could not load symbols."]}

def get_yfinance_data(symbol, timeframe):
    timeframe_map = {
        '15m': {'period': '5d', 'interval': '15m'}, '1h': {'period': '1mo', 'interval': '60m'},
        '4h': {'period': '3mo', 'interval': '90m'}, '1d': {'period': '1y', 'interval': '1d'},
        '1w': {'period': '5y', 'interval': '1wk'},
    }
    params = timeframe_map.get(timeframe, {'period': '1y', 'interval': '1d'})

    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        hist = ticker.history(period=params['period'], interval=params['interval'], auto_adjust=False)

        if hist.empty: return {"error": f"No data for {symbol} at {timeframe} timeframe."}
        
        # Calculate indicators using pandas-ta
        with pd.option_context('mode.chained_assignment', None):
            hist.ta.strategy(ta.Strategy(
                name="Comprehensive",
                ta=[
                    {"kind": "rsi"}, {"kind": "macd"}, {"kind": "bbands"},
                    {"kind": "sma", "length": 50}, {"kind": "sma", "length": 100},
                    {"kind": "stochrsi"}, {"kind": "adx"}, {"kind": "vwap"},
                    {"kind": "atr"}, {"kind": "ema", "length": 20}, {"kind": "obv"},
                    {"kind": "willr"}, {"kind": "supertrend"}
                ]
            ))
        
        # Manual pivot calculation
        prev_day = hist.iloc[-2]
        ph, pl, pc = prev_day['High'], prev_day['Low'], prev_day['Close']
        pivot = (ph + pl + pc) / 3
        r1, s1 = (2 * pivot) - pl, (2 * pivot) - ph
        r2, s2 = pivot + (ph - pl), pivot - (ph - pl)
        
        # Manual Fibonacci calculation for the last 30 periods
        recent_period = hist.iloc[-30:]
        fib_high = recent_period['High'].max()
        fib_low = recent_period['Low'].min()

        latest = hist.iloc[-1].to_dict()
        data = {
            'symbol': info.get('symbol').replace('.NS', ''), 'name': info.get('longName'), 'price': latest['Close'],
            'previous_close': pc, 'volume': latest['Volume'], 'marketCap': info.get('marketCap'),
            'high52': info.get('fiftyTwoWeekHigh'), 'low52': info.get('fiftyTwoWeekLow'), 'pe': info.get('trailingPE'),
            'divYield': info.get('dividendYield', 0) * 100, 'beta': info.get('beta'),
            'change': ((latest['Close'] - pc) / pc) * 100,
            'change_1m': ((latest['Close'] - hist['Close'].iloc[-22]) / hist['Close'].iloc[-22]) * 100 if len(hist) > 22 else 0,
        }
        
        data['all_indicators'] = {
            'rsi': latest.get('RSI_14'), 'macd': latest.get('MACD_12_26_9'), 'macd_signal': latest.get('MACDs_12_26_9'),
            'macd_hist': latest.get('MACDh_12_26_9'), 'bb_upper': latest.get('BBU_20_2.0'), 'bb_lower': latest.get('BBL_20_2.0'),
            'sma50': latest.get('SMA_50'), 'sma100': latest.get('SMA_100'),
            'stochrsi_k': latest.get('STOCHRSIk_14_14_3_3'), 'stochrsi_d': latest.get('STOCHRSId_14_14_3_3'),
            'adx': latest.get('ADX_14'), 'vwap': latest.get('VWAP_D'), 'atr': latest.get('ATRr_14'),
            'ema': latest.get('EMA_20'), 'obv': latest.get('OBV'),
            'williams': latest.get('WILLR_14'), 'supertrend_dir': latest.get('SUPERTd_7_3.0'), 'supertrend_val': latest.get('SUPERT_7_3.0'),
            'pivot': pivot, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2,
            'fib_high': fib_high, 'fib_low': fib_low
        }
        return data
    except Exception as e:
        return {"error": f"Data calculation failed for {symbol}. Error: {e}"}

def get_gemini_analysis(stock_data, risk_tolerance):
    if not gemini_model: return {"error": "Gemini API is unavailable."}
    prompt = f"""
    You are StockSense Pro, an expert financial analyst AI for a '{risk_tolerance}' risk investor. Analyze {stock_data['name']}.

    **Data:** Price: {stock_data['price']:.2f}, P/E: {stock_data.get('pe', 'N/A')}, RSI: {stock_data['all_indicators'].get('rsi', 0):.2f}, ADX: {stock_data['all_indicators'].get('adx', 0):.2f}

    **Tasks:** Fulfill all sections with structured, point-wise insights.
    1.  **Strategic Outlook:** A summary paragraph and a bulleted list of 2-3 key fundamental/sentiment takeaways.
    2.  **Tactical Analysis:** A summary paragraph and a bulleted list of 2-3 key technical takeaways.
    3.  **AI Thought Process:** 6 insightful observations (mix of fundamental/technical).
    4.  **Conceptual Matrices:** Values for: volatility, profit_probability, risk_reward_ratio, historical_accuracy.
    5.  **Predictive Insights:** All 4 parts: Momentum Forecast (text, probability), Sentiment & Risk (text, risk_level), Price Targets (short, medium, long_term, downside_risk), Risk Assessment (factors list).
    6.  **Trade Strategy:** A recommendation, Entry, Stop Loss, TP1, TP2.

    **Output ONLY a single valid JSON object.**
    {{
      "strategic_outlook": "Summary paragraph...\\n\\n* Takeaway 1...\\n* Takeaway 2...",
      "tactical_analysis": "Summary paragraph...\\n\\n* Takeaway 1...\\n* Takeaway 2...",
      "thought_process": ["...", "...", "...", "...", "...", "..."],
      "conceptual_matrices": {{"volatility": "...", "profit_probability": "...", "risk_reward_ratio": "...", "historical_accuracy": "..."}},
      "predictive_insights": {{
        "momentum_forecast": {{"text": "...", "probability": 85}},
        "market_sentiment": {{"text": "...", "risk_level": "Medium"}},
        "price_targets": {{"short_term": "{stock_data['price'] * 1.05:.0f}", "medium_term": "{stock_data['price'] * 1.12:.0f}", "long_term": "{stock_data['price'] * 1.20:.0f}", "downside_risk": "{stock_data['price'] * 0.95:.0f}"}},
        "risk_assessment": {{"factors": ["...", "..."]}}
      }},
      "trade_strategy": {{"recommendation": "...", "entry": "{stock_data['price'] * 0.99:.0f}", "stop_loss": "{stock_data['price'] * 0.96:.0f}", "tp1": "{stock_data['price'] * 1.06:.0f}", "tp2": "{stock_data['price'] * 1.15:.0f}"}}
    }}
    """
    try:
        response = gemini_model.generate_content(prompt)
        json_text = re.search(r'\{[\s\S]*\}', response.text).group(0)
        return json.loads(json_text)
    except Exception as e:
        return {"error": f"An error occurred with the Gemini API: {e}"}

# --- Flask Routes ---
@app.route('/')
def dashboard():
    return render_template('Dashboard.html', stock_sectors=get_stock_symbols())

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        payload = request.get_json()
        symbol, risk, timeframe = payload.get('stock'), payload.get('risk'), payload.get('timeframe')
        if not symbol: return jsonify({"error": "Stock symbol not provided."}), 400

        stock_data = get_yfinance_data(symbol, timeframe)
        if "error" in stock_data: return jsonify(stock_data), 404

        ai_analysis = get_gemini_analysis(stock_data, risk)
        
        final_response = { **stock_data, "ai_analysis": ai_analysis }
        sanitized_response = convert_numpy_types(final_response)
        
        return jsonify(sanitized_response)
    except Exception as e:
        return jsonify({"error": "A critical server error occurred. Check logs."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=5000, debug=True)