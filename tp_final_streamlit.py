
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Análisis de Cancelaciones Delivery", layout="wide")

st.title("📊 Introducción a Ciencia de Datos - Análisis de Cancelaciones")

uploaded = st.file_uploader("Cargar dataset CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.sidebar.header("Filtros")
    if "restaurant_city" in df.columns:
        ciudades = st.sidebar.multiselect(
            "Ciudad",
            sorted(df["restaurant_city"].dropna().unique()),
            default=sorted(df["restaurant_city"].dropna().unique())
        )
        df = df[df["restaurant_city"].isin(ciudades)]

    st.header("Vista General")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pedidos", len(df))

    if "order_status" in df.columns:
        cancelados = (df["order_status"] == "cancelled").sum()
        tasa = cancelados / len(df) * 100
        c2.metric("Cancelados", cancelados)
        c3.metric("Tasa cancelación", f"{tasa:.2f}%")

    st.subheader("Muestra")
    st.dataframe(df.head())

    st.header("Calidad de Datos")

    nulls = df.isna().sum()
    st.dataframe(
        pd.DataFrame({
            "Columna": nulls.index,
            "Nulos": nulls.values,
            "%": (nulls.values/len(df)*100).round(2)
        }).sort_values("Nulos", ascending=False)
    )

    st.header("EDA de Cancelaciones")

    if "order_status" in df.columns:
        fig, ax = plt.subplots()
        df["order_status"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    tabs = st.tabs([
        "Ciudad",
        "Categorías",
        "Canales",
        "Clima",
        "Temperatura"
    ])

    with tabs[0]:
        if {"restaurant_city","order_status"}.issubset(df.columns):
            tmp = df.groupby("restaurant_city").agg(
                pedidos=("order_status","count"),
                cancelados=("order_status",lambda x:(x=="cancelled").sum())
            )
            tmp["tasa"] = tmp["cancelados"]/tmp["pedidos"]*100
            st.dataframe(tmp.sort_values("tasa", ascending=False))

            fig, ax = plt.subplots(figsize=(8,4))
            tmp["tasa"].sort_values().plot.barh(ax=ax)
            st.pyplot(fig)

    with tabs[1]:
        if {"menu_item_category","order_status"}.issubset(df.columns):
            tmp = df.groupby("menu_item_category").agg(
                pedidos=("order_status","count"),
                cancelados=("order_status",lambda x:(x=="cancelled").sum())
            )
            tmp["tasa"] = tmp["cancelados"]/tmp["pedidos"]*100

            fig, ax = plt.subplots(figsize=(8,4))
            tmp["tasa"].sort_values().plot.barh(ax=ax)
            st.pyplot(fig)

    with tabs[2]:
        if {"customer_acquisition_channel","order_status"}.issubset(df.columns):
            tmp = df.groupby("customer_acquisition_channel").agg(
                pedidos=("order_status","count"),
                cancelados=("order_status",lambda x:(x=="cancelled").sum())
            )
            tmp["tasa"] = tmp["cancelados"]/tmp["pedidos"]*100

            fig, ax = plt.subplots(figsize=(8,4))
            tmp["tasa"].sort_values().plot.barh(ax=ax)
            st.pyplot(fig)

    with tabs[3]:
        if {"weather_precipitation","order_status"}.issubset(df.columns):
            lluvia = df[df["weather_precipitation"] > 0]
            st.write("Pedidos con precipitaciones:", len(lluvia))

            if len(lluvia):
                fig, ax = plt.subplots()
                lluvia["order_status"].value_counts().plot.pie(
                    autopct="%1.1f%%", ax=ax
                )
                ax.set_ylabel("")
                st.pyplot(fig)

    with tabs[4]:
        if {"weather_temperature","order_status"}.issubset(df.columns):
            fig, ax = plt.subplots()
            df.boxplot(column="weather_temperature",
                       by="order_status",
                       ax=ax)
            st.pyplot(fig)

    st.header("Variables Numéricas")
    st.dataframe(df.describe())

else:
    st.info("Cargá el CSV utilizado en el notebook para comenzar.")
