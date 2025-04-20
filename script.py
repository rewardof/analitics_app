import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import requests


# Fetch data dynamically from the World Bank API
def fetch_world_bank_data(indicator, country="UZ", start_year=2010, end_year=2024):
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
        df = pd.DataFrame(records).dropna()
        df.sort_values(by="Yil", inplace=True)
        return df
    else:
        st.error(f"Failed to fetch data for indicator {indicator}. Status code: {response.status_code}")
        return None


# Combine fetched data into a single DataFrame
def load_dynamic_data():
    indicators = {
        "NY.GDP.MKTP.CD": "YIM",
        "FP.CPI.TOTL.ZG": "Inflyatsiya",
        "SL.UEM.TOTL.ZS": "Ishsizlik"
    }
    data_frames = []
    for code, name in indicators.items():
        df = fetch_world_bank_data(code)
        if df is not None:
            df.rename(columns={code: name}, inplace=True)
            data_frames.append(df)

    combined_df = data_frames[0]
    for df in data_frames[1:]:
        combined_df = pd.merge(combined_df, df, on="Yil", how="outer")

    combined_df.sort_values(by="Yil", inplace=True)
    return combined_df


# Forecast using ARIMA
def forecast(data, column, years=6):
    model = ARIMA(data[column].dropna(), order=(2, 1, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=years)
    return forecast


# Streamlit UI
st.title("O‘zbekiston iqtisodiy ko‘rsatkichlari (2010–2024 + prognoz 2025–2030)")

data = load_dynamic_data()

if data is not None and not data.empty:
    st.write("### Asosiy ma'lumotlar (2010–2024)")
    st.dataframe(data)

    yim_chart = px.line(data, x="Yil", y="YIM", title="Yalpi Ichki Mahsulot (YIM) o‘sishi", markers=True)
    st.plotly_chart(yim_chart)

    fig = px.line(data, x="Yil", y=["Inflyatsiya", "Ishsizlik"],
                  title="Inflyatsiya va Ishsizlik darajasi", markers=True)
    st.plotly_chart(fig)

    # Forecast years
    forecast_years = 6
    future_years = list(range(2025, 2031))

    # YIM Forecast
    st.write("### YIM prognozi (2025–2030)")
    forecast_yim = forecast(data, "YIM", years=forecast_years)
    forecast_yim_df = pd.DataFrame({"Yil": future_years, "YIM": forecast_yim})
    full_yim_data = pd.concat([data[["Yil", "YIM"]], forecast_yim_df])
    full_yim_data["Turi"] = ["Haqiqiy" if yil <= 2024 else "Prognoz" for yil in full_yim_data["Yil"]]
    forecast_yim_chart = px.line(full_yim_data, x="Yil", y="YIM", color="Turi",
                                 title="YIM o‘sishi va prognozi", markers=True)
    st.plotly_chart(forecast_yim_chart)

    # Inflyatsiya Forecast
    st.write("### Inflyatsiya prognozi (2025–2030)")
    forecast_inflation = forecast(data, "Inflyatsiya", years=forecast_years)
    forecast_inflation_df = pd.DataFrame({"Yil": future_years, "Inflyatsiya": forecast_inflation})
    full_inflation_data = pd.concat([data[["Yil", "Inflyatsiya"]], forecast_inflation_df])
    full_inflation_data["Turi"] = ["Haqiqiy" if yil <= 2024 else "Prognoz" for yil in full_inflation_data["Yil"]]
    forecast_inflation_chart = px.line(full_inflation_data, x="Yil", y="Inflyatsiya", color="Turi",
                                       title="Inflyatsiya darajasi va prognozi", markers=True)
    st.plotly_chart(forecast_inflation_chart)

    # Ishsizlik Forecast
    st.write("### Ishsizlik prognozi (2025–2030)")
    forecast_unemployment = forecast(data, "Ishsizlik", years=forecast_years)
    forecast_unemployment_df = pd.DataFrame({"Yil": future_years, "Ishsizlik": forecast_unemployment})
    full_unemployment_data = pd.concat([data[["Yil", "Ishsizlik"]], forecast_unemployment_df])
    full_unemployment_data["Turi"] = ["Haqiqiy" if yil <= 2024 else "Prognoz" for yil in full_unemployment_data["Yil"]]
    forecast_unemployment_chart = px.line(full_unemployment_data, x="Yil", y="Ishsizlik", color="Turi",
                                          title="Ishsizlik darajasi va prognozi", markers=True)
    st.plotly_chart(forecast_unemployment_chart)
else:
    st.error("Ma'lumotlarni yuklashda xatolik yuz berdi.")
