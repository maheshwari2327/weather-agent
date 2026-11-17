"""Weather Prediction Agent — Streamlit Web Application.

Run:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd

from weather_agent import WeatherPredictionAgent
from weather_service import (
    search_city,
    fetch_weather_data,
    calculate_temperature_trend,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Weather Prediction Agent",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #BFDBFE;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1E40AF;
        margin-bottom: 0.5rem;
    }
    .prediction-reasoning {
        font-size: 1.05rem;
        color: #1F2937;
        line-height: 1.5;
        margin-bottom: 0.75rem;
    }
    .forecast-line {
        font-size: 1.0rem;
        font-weight: 600;
        color: #047857;
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Agent Settings")
    ollama_host = st.text_input("Ollama Host URL", value=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model = st.text_input("Model Name", value=os.getenv("OLLAMA_MODEL", "llama3.2"))
    model_temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    agent = WeatherPredictionAgent(
        model_name=ollama_model,
        ollama_host=ollama_host,
        temperature=model_temp,
    )

    st.markdown("---")
    st.subheader("🔌 Ollama Connection")
    if agent.is_ollama_available():
        st.success(f"🟢 Connected to `{ollama_model}`")
    else:
        st.warning("🟠 Ollama server is offline")
        st.caption("Agent will use the built-in mathematical regression model as fallback.")
        with st.expander("How to start Ollama"):
            st.code("ollama pull llama3.2\nollama serve", language="bash")

    st.markdown("---")
    st.subheader("📖 How It Works")
    st.markdown(
        """
        1. **Look up past weather**: Pulls 7 days of actual max temps from Open-Meteo.
        2. **Calculate trend**: Finds the daily rate of change (°C/day) & recent highs/lows.
        3. **AI Prediction**: Local model reasons through the pattern to forecast tomorrow's high.
        4. **Official Comparison**: Compares against the website's official tomorrow forecast.
        """
    )

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🌤️ Weather Prediction Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Predicts <b>tomorrow\'s maximum temperature</b> for any city worldwide by analyzing 7-day historical trends.</div>',
    unsafe_allow_html=True,
)

# Preset city quick selection
preset_cities = ["Hyderabad", "London", "Tokyo", "New York", "Mumbai", "Paris", "Nairobi"]
selected_preset = st.pills("Quick pick a city:", preset_cities, default=None)

col_input, col_btn = st.columns([4, 1])
with col_input:
    city_input = st.text_input(
        "City Name",
        value=selected_preset if selected_preset else "Hyderabad",
        placeholder="Type a city name (e.g. Hyderabad, London, Tokyo)...",
        label_visibility="collapsed",
    )
with col_btn:
    predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Prediction Execution
# ---------------------------------------------------------
if predict_clicked or (selected_preset and not st.session_state.get("_last_preset") == selected_preset):
    st.session_state["_last_preset"] = selected_preset
    city_query = city_input.strip()

    if not city_query:
        st.warning("Please enter a valid city name.")
    else:
        with st.status(f"Analyzing weather for **{city_query}**...", expanded=True) as status:
            # Step 1: Lookup city
            status.write(f"🔍 Thinking... looking up past temperatures for **{city_query}**")
            city_info = search_city(city_query)

            if not city_info:
                status.update(label="City Not Found", state="error", expanded=True)
                st.error(f"❌ City **'{city_query}'** wasn't found. Please check spelling.")
                st.stop()

            resolved_city = city_info["name"]
            country = city_info.get("country", "")

            # Step 2: Fetch 7-day data + forecast
            weather_data = fetch_weather_data(
                latitude=city_info["latitude"],
                longitude=city_info["longitude"],
                timezone=city_info.get("timezone", "auto"),
            )

            if not weather_data:
                status.update(label="Weather Service Error", state="error", expanded=True)
                st.error("❌ Could not fetch weather data. Please check your internet connection.")
                st.stop()

            past_records = weather_data.get("past_7_days", [])
            tomorrow = weather_data.get("tomorrow_forecast", {})
            past_max_temps = [r["max_temp"] for r in past_records if r.get("max_temp") is not None]

            # Step 3: Calculate trend
            status.write("📈 Thinking... calculating the temperature trend")
            trend = calculate_temperature_trend(past_max_temps)

            # Step 4: AI prediction
            status.write("🧠 Thinking... generating reasoned temperature prediction")
            ollama_online = agent.is_ollama_available()
            if ollama_online:
                prediction_text = agent._generate_ai_prediction(
                    city_name=resolved_city,
                    country=country,
                    past_temps=past_max_temps,
                    trend=trend,
                )
                source_label = f"Ollama ({ollama_model})"
            else:
                prediction_text = agent._generate_analytical_prediction(
                    city_name=resolved_city,
                    trend=trend,
                )
                source_label = "Statistical Regression Engine"

            status.update(label=f"Done! Weather analyzed for {resolved_city}, {country}", state="complete", expanded=False)

        # ---------------------------------------------------------
        # Display Results
        # ---------------------------------------------------------
        st.markdown(f"### 📍 Weather Forecast: **{resolved_city}, {country}**")

        # Metric Cards
        official_max = tomorrow.get("max_temp")
        official_min = tomorrow.get("min_temp")
        predicted_max = round(trend["projected_temp"])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                label="Predicted Tomorrow Max",
                value=f"{predicted_max:.0f}°C",
                delta=f"{trend['slope']:+.1f}°C/day ({trend['trend_direction']})",
            )
        with m2:
            st.metric(
                label="Official Tomorrow Max",
                value=f"{official_max:.1f}°C" if official_max is not None else "N/A",
                delta=f"Min: {official_min:.1f}°C" if official_min is not None else None,
                delta_color="off",
            )
        with m3:
            st.metric(
                label="Past 7-Day High / Low",
                value=f"{trend['max_temp']:.1f}°C",
                delta=f"Low: {trend['min_temp']:.1f}°C",
                delta_color="off",
            )
        with m4:
            st.metric(
                label="Past 7-Day Average",
                value=f"{trend['avg_temp']:.1f}°C",
                delta=f"Recent: {trend['recent_temp']:.1f}°C",
                delta_color="off",
            )

        # Prediction Highlight Card
        lines = prediction_text.strip().split("\n", 1)
        pred_header = lines[0] if len(lines) > 0 else ""
        pred_explanation = lines[1] if len(lines) > 1 else ""

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-title">{pred_header}</div>
                <div class="prediction-reasoning">{pred_explanation}</div>
                <div class="forecast-line">
                    Official forecast for tomorrow → max {official_max:.1f}°C, min {official_min:.1f}°C
                </div>
                <div style="margin-top: 0.6rem; font-size: 0.85rem; color: #6B7280;">
                    AI Engine: <b>{source_label}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------------------------------------------
        # Historical Trend Chart
        # ---------------------------------------------------------
        st.subheader("📊 7-Day Temperature Trend & Projection")

        chart_data = []
        for r in past_records:
            chart_data.append({
                "Date": r["date"],
                "Actual Max Temp (°C)": r["max_temp"],
                "Actual Min Temp (°C)": r.get("min_temp"),
            })

        # Append tomorrow's projected point
        chart_data.append({
            "Date": f"{tomorrow.get('date', 'Tomorrow')} (Forecast)",
            "Actual Max Temp (°C)": official_max,
            "Actual Min Temp (°C)": official_min,
        })

        df = pd.DataFrame(chart_data).set_index("Date")
        st.line_chart(df, height=320)

        # Historical Data Table in Expander
        with st.expander("📋 View Raw 7-Day Temperature Records"):
            records_df = pd.DataFrame(past_records)
            records_df.columns = ["Date", "Max Temperature (°C)", "Min Temperature (°C)"]
            st.dataframe(records_df, use_container_width=True)
