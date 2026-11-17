# 🌤️ Weather Prediction Agent

An intelligent weather prediction agent that forecasts **tomorrow's maximum temperature** for any city worldwide by analyzing actual temperature trends from the past 7 days.

---

## 📌 Overview

Traditional weather forecasts rely on complex numerical simulation models. This agent acts like a hands-on meteorological analyst:
1. **Pulls 7-day historical records** using the free Open-Meteo API (no API key required).
2. **Calculates temperature trend slope**, identifying whether it is heating up, cooling down, or holding steady.
3. **Uses a local AI brain (Ollama / Llama 3.2)** to reason through the pattern and predict tomorrow's high.
4. **Compares with the official forecast** so you can see how closely the trend-based AI prediction aligns with actual forecasts.

---

## 🌐 How to Run in Website (Web Browser)

You can launch a interactive web UI using Streamlit:

```bash
streamlit run app.py
```

Once launched, your default web browser will automatically open:
```
Local URL: http://localhost:8501
```

### Web App Features:
- 🔍 **City Search Bar & Quick Pick Buttons** (Hyderabad, London, Tokyo, New York, Mumbai, Paris, Nairobi).
- 🔄 **Live Status Steps** showing data fetching, trend calculation, and AI thinking.
- 📊 **Interactive 7-Day Temperature Chart** comparing historical temperatures with the tomorrow projection.
- 🏷️ **Side-by-side Cards** showing Predicted vs Official Forecast with delta differences.
- ⚙️ **Sidebar Settings** allowing you to customize Ollama Host, Model, and Temperature with live connection status.

---

## 💻 How to Run in Terminal (CLI)

### Interactive Mode:
```bash
python main.py
```

Type any city name:
```text
City: Hyderabad
Thinking... looking up past temperatures for Hyderabad
Thinking... calculating the temperature trend

Prediction: Tomorrow's maximum will be around 31°C.
The past week shows a slight downward trend of -0.3°C per day,
suggesting a mild cool-down from the recent high of 32°C.

Official forecast for tomorrow → max 31.2°C, min 20.8°C
```

### Single Command Mode:
```bash
python main.py "Tokyo"
```

---

## 📂 Project Structure

```
weather agent/
├── app.py                   # 🌐 Streamlit Web Application
├── main.py                  # 💻 Terminal CLI Entry Point
├── weather_agent.py         # WeatherPredictionAgent class (AI reasoning + fallback)
├── weather_service.py       # Open-Meteo API fetching & trend math
├── tools.py                 # LangChain tool definitions (@tool)
├── prompt_nontechnical.md   # Plain English guide & user specification
├── test_agent.py            # Unit & mock tests
├── requirements.txt         # Python dependencies
├── .env.example             # Ollama configuration template
└── README.md                # Project documentation
```

---

## 🛠️ Setup & Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Run Local AI with Ollama
If you want the AI brain to reason about the temperatures locally:
1. Install [Ollama](https://ollama.com).
2. Download the model:
   ```bash
   ollama pull llama3.2
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```
*(Note: If Ollama is not running, the agent automatically falls back to its analytical trend model without crashing!)*

---

## 🧪 Testing

Run automated tests:
```bash
python -m unittest test_agent.py
```
