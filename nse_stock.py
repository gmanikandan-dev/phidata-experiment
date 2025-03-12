from fastapi import FastAPI, Query
import uvicorn
from groq import Groq
from nsetools import Nse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Set up NSE & Groq Clients
nse = Nse()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def get_stock_price(symbol: str):
    """Fetch stock price from NSE India"""
    try:
        stock_info = nse.get_quote(symbol)
        return {
            "symbol": stock_info["symbol"],
            "last_price": stock_info["lastPrice"],
            "change": stock_info["change"],
            "percentage_change": stock_info["pChange"],
            "high": stock_info["dayHigh"],
            "low": stock_info["dayLow"],
            "previous_close": stock_info["previousClose"],
        }
    except Exception as e:
        return {"error": f"Stock data not found: {str(e)}"}

def analyze_stock_with_groq(stock_data: dict):
    """Analyze stock trends using Groq LLM"""
    try:
        stock_summary = (
            f"Stock: {stock_data['symbol']}\n"
            f"Last Price: {stock_data['last_price']} INR\n"
            f"Change: {stock_data['change']} ({stock_data['percentage_change']}%)\n"
            f"High: {stock_data['high']}, Low: {stock_data['low']}\n"
            f"Previous Close: {stock_data['previous_close']}"
        )

        response = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL"),
            messages=[{"role": "user", "content": f"Analyze the following stock:\n\n{stock_summary}"}],
        )

        return response.choices[0].message.content if response.choices else "No analysis available"
    except Exception as e:
        return f"Groq API Error: {str(e)}"

@app.get("/stock/")
def get_stock(symbol: str, analyze: bool = Query(False)):
    """Retrieve and analyze Indian stock market data."""
    stock_data = get_stock_price(symbol.upper())

    if "error" in stock_data:
        return stock_data

    analysis = analyze_stock_with_groq(stock_data) if analyze else None
    return {"stock_data": stock_data, "analysis": analysis}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
