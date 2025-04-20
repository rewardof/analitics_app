import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import requests


# Fetch data dynamically from the World Bank API
def fetch_world_bank_data(indicator, country="UZ", start_year=2010, end_year=2023):
    url = f"http://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start_year}:{end_year}&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Extract relevant data
        records = []
        for item in data[1]:
            year = int(item['date'])
            value = item['value']
            records.append({"Yil": year, indicator: value})
        df = pd.DataFrame(records).dropna()  # Drop missing values
        df.sort_values(by="Yil", inplace=True)  # Sort by year
        return df
    else:
        st.error(f"Failed to fetch data for indicator {indicator}. Status code: {response.status_code}")
        return None


# Combine fetched data into a single DataFrame
def load_dynamic_data():
    # Define indicators (World Bank codes)
    indicators = {
        "NY.GDP.MKTP.KD.ZG": "YIM",  # GDP growth (annual %)
        "FP.CPI.TOTL.ZG": "Inflyatsiya",  # Inflation, consumer prices (annual %)
        "SL.UEM.TOTL.ZS": "Ishsizlik"  # Unemployment rate (% of total labor force)
    }
    data_frames = []
    for code, name in indicators.items():
        df = fetch_world_bank_data(code)
        if df is not None:
            df.rename(columns={code: name}, inplace=True)
            data_frames.append(df)

    # Merge all data into one DataFrame
    combined_df = data_frames[0]
    for df in data_frames[1:]:
        combined_df = pd.merge(combined_df, df, on="Yil", how="outer")

    combined_df.sort_values(by="Yil", inplace=True)
    return combined_df


# Forecast using ARIMA
def forecast(data, column, years=5):
    model = ARIMA(data[column].dropna(), order=(2, 1, 2))  # ARIMA(p, d, q)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=years)
    return forecast


# Streamlit UI
st.title("O‘zbekiston iqtisodiy ko‘rsatkichlari")

# Load dynamic data
data = load_dynamic_data()

if data is not None and not data.empty:
    st.write("### Asosiy ma'lumotlar")
    st.dataframe(data)

    # Vizualizatsiya: YIM o‘sishi
    yim_chart = px.line(data, x="Yil", y="YIM", title="Yalpi Ichki Mahsulot (YIM) o‘sishi", markers=True)
    st.plotly_chart(yim_chart)

    # Vizualizatsiya: Inflyatsiya va Ishsizlik
    fig = px.line(data, x="Yil", y=["Inflyatsiya", "Ishsizlik"],
                  title="Inflyatsiya va Ishsizlik darajasi", markers=True)
    st.plotly_chart(fig)

    # Prognoz qilish
    st.write("### YIM prognozi (2024-2028)")
    forecast_years = 5
    forecast_data = forecast(data, "YIM", years=forecast_years)
    future_years = list(range(data["Yil"].max() + 1, data["Yil"].max() + 1 + forecast_years))
    forecast_df = pd.DataFrame({
        "Yil": future_years,
        "YIM": forecast_data
    })

    # Combine actual and forecasted data
    full_data = pd.concat([data[["Yil", "YIM"]], forecast_df])
    full_data["Turi"] = ["Haqiqiy" if yil <= data["Yil"].max() else "Prognoz" for yil in full_data["Yil"]]

    # Plot forecasted data
    forecast_chart = px.line(full_data, x="Yil", y="YIM", color="Turi",
                             title="YIM o‘sishi va prognozi", markers=True)
    st.plotly_chart(forecast_chart)
else:
    st.error("Ma'lumotlarni yuklashda xatolik yuz berdi.")