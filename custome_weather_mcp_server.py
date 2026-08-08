# -----------------------------------------------------------------------------------
#                                   Import & Init Statements
# -----------------------------------------------------------------------------------

from mcp.server.fastmcp import FastMCP
import requests
import os 
from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("Weather MCP Server")

OPEN_WEATHER_API = os.getenv("OPEN_WEATHER_API")


# -----------------------------------------------------------------------------------
#                                   Logic Statements
# -----------------------------------------------------------------------------------

@mcp.tool()
def get_current_weather(city : str):
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params= {
            "q" : city,
            "appid" : OPEN_WEATHER_API,
            "units" : "metric"
        }
    )
    data = response.json()

    if response.status_code != 200:
        return data 

    return {
        "city" : data['name'],
        "temperature_c" : data['main']['temp'],
        "feels_like_c" : data['main']['feels_like'],
        "humidity" : data['main']['humidity'],
        "condition" : data['weather'][0]['description'],
        "wind_speed" : data['wind']['speed']
    }

@mcp.tool()
def get_forecast(city : str):
    url = (
        "https://api.openweathermap.org/data/2.5/forecast"
    )
    params = {
        "q" : city,
        "appid" : OPEN_WEATHER_API,
        "units" : "metric"
    }
    response = requests.get(
        url , params=params 
    )
    data = response.json()
    forecast = []

    for item  in data['list'][:5]:
        forecast.append({
            "datetime" : item['dt_txt'],
            "temperature" : item['main']['temp'],
            "weather" : item['weather'][0]['description']
        })

    return {
        "city" : city,
        "forecast" : forecast
    }


if __name__ == "__main__":
    mcp.run()