import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA


# Ma'lumotlarni yuklash (namuna ma'lumotlar)
def load_data():
    data = {
        "Yil": list(range(2010, 2025)),
        "YIM": [34.2, 37.5, 41.0, 45.2, 50.1, 55.5, 61.3, 67.8, 74.0, 80.5, 88.0, 96.5, 106.0, 116.7, 128.2],
        "Inflyatsiya": [7.8, 8.1, 8.5, 8.9, 9.2, 9.7, 10.2, 11.0, 11.8, 12.5, 13.3, 14.0, 14.8, 15.5, 16.2],
        "Ishsizlik": [9.5, 9.3, 9.0, 8.8, 8.6, 8.4, 8.2, 8.0, 7.8, 7.5, 7.3, 7.1, 6.9, 6.7, 6.5]
    }
    return pd.DataFrame(data)


def forecast(data, column, years=5):
    model = ARIMA(data[column], order=(2, 1, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=years)
    return forecast


# Streamlit UI
st.title("O‘zbekiston iqtisodiy ko‘rsatkichlari")

data = load_data()
st.write("### Asosiy ma'lumotlar")
st.dataframe(data)

# Vizualizatsiya: YIM o‘sishi
yim_chart = px.line(data, x="Yil", y="YIM", title="Yalpi Ichki Mahsulot (YIM) o‘sishi", markers=True)
st.plotly_chart(yim_chart)

# Vizualizatsiya: Inflyatsiya va Ishsizlik
fig = px.line(data, x="Yil", y=["Inflyatsiya", "Ishsizlik"], title="Inflyatsiya va Ishsizlik darajasi", markers=True)
st.plotly_chart(fig)

# Prognoz qilish
st.write("### YIM prognozi (5 yil) ")
forecast_data = forecast(data, "YIM")
st.line_chart(np.concatenate((data["YIM"].values, forecast_data)))
