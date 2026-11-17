"""Weather Prediction Agent — Main CLI Entry Point.

Run:
    python main.py
    python main.py Hyderabad
"""

import sys
from weather_agent import WeatherPredictionAgent


def print_banner():
    print("=" * 60)
    print("        🌤️   Weather Prediction Agent — Day 17")
    print("=" * 60)
    print("Predicts tomorrow's maximum temperature from 7-day trends.")
    print("Type a city name (e.g. Hyderabad, London, Tokyo, New York).")
    print("Type 'exit' or 'quit' to close the program.\n")


def run_prediction_for_city(agent: WeatherPredictionAgent, city_name: str):
    result = agent.predict(city_name=city_name, show_thinking=True)
    if not result.get("success"):
        print(f"\n❌ {result.get('error')}\n")
        return

    print(f"\n{result['prediction_text']}\n")
    print(f"{result['forecast_line']}\n")


def main():
    agent = WeatherPredictionAgent()

    # If city is supplied as command-line argument
    if len(sys.argv) > 1:
        city = " ".join(sys.argv[1:]).strip()
        print(f"City: {city}")
        run_prediction_for_city(agent, city)
        return

    # Interactive loop
    print_banner()

    # Notice if Ollama is online or offline
    if not agent.is_ollama_available():
        print("💡 Note: Local Ollama server not detected (http://localhost:11434).")
        print("   Running in statistical trend mode. To use local LLM: 'ollama run llama3.2'\n")

    while True:
        try:
            city = input("City: ").strip()
            if not city:
                continue
            if city.lower() in ("exit", "quit", "q"):
                print("Goodbye! Stay weather-ready! 🌤️")
                break

            run_prediction_for_city(agent, city)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye! 🌤️")
            break


if __name__ == "__main__":
    main()
