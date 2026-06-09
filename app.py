import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gdown
import math
import pathlib
import warnings

warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Introducción a Ciencia de Datos",
    page_icon="🛵",
    layout="wide",
)

# ─── Header / portada ───────────────────────────────────────────────────────
st.title("🛵 Introducción a Ciencia de Datos")
st.markdown(
    """
    **Carrera:** Tecnicatura universitaria en Ciencia de Datos e IA  
    **Año:** 2026  
    **Integrantes:** Deheza, Julian · Gargiulo, Franco · Girondo, Mauro · Tusoli, Melisa
    """
)
st.divider()

# ─── Dataset loader (cached) ────────────────────────────────────────────────
@st.cache_data(show_spinner="Descargando dataset desde Google Drive…")
def load_data():
    file_id = "1ftNIrya9c9B_smlUSsRlyfqciRmov3ha"
    output_filename = "downloaded_data.csv"
    gdown.download(id=file_id, output=output_filename, quiet=True)
    df = pd.read_csv(output_filename)
    df.index += 1
    return df

df = load_data()

# ─── SECCIÓN 1: Exploración inicial ─────────────────────────────────────────
st.header("1. Primeras exploraciones del dataset")

with st.expander("📋 Muestra del dataset (head)", expanded=True):
    st.dataframe(df.head(), use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    with st.expander("🔢 Info general (dtypes)"):
        buf = pd.io.common.BytesIO() if False else None
        dtype_df = pd.DataFrame({"dtype": df.dtypes.astype(str)})
        st.dataframe(dtype_df, use_container_width=True)

with col2:
    with st.expander("📊 Describe numérico"):
        st.dataframe(df.describe().round(2), use_container_width=True)

with st.expander("🔍 Describe completo (incluye categóricos)"):
    st.dataframe(df.describe(include="all"), use_container_width=True)

# Valores únicos
with st.expander("🧮 Valores únicos por columna"):
    summary = df.nunique().reset_index()
    summary.columns = ["columna", "unicos"]
    summary["ratio"] = (summary["unicos"] / len(df) * 100).round(2)
    st.dataframe(summary, use_container_width=True)

# Duplicados
with st.expander("🔁 Análisis de duplicados"):
    dup_rows = df.duplicated().sum()
    dup_orders = df["order_id"].duplicated().sum() if "order_id" in df.columns else "N/A"
    st.metric("Filas duplicadas", dup_rows)
    st.metric("Órdenes duplicadas (order_id)", dup_orders)
    if dup_rows > 0:
        st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)
    else:
        st.success("No se encontraron filas duplicadas.")

# Nulls
with st.expander("🚫 Valores nulos"):
    nulls = df.isna().sum()
    nulls_pct = (df.isna().mean() * 100).round(2)
    nulls_df = pd.DataFrame({"nulls": nulls, "pct": nulls_pct}).sort_values("pct", ascending=False)
    nulls_con = nulls_df[nulls_df["nulls"] > 0]
    if len(nulls_con) > 0:
        st.dataframe(nulls_con, use_container_width=True)
    else:
        st.success("No se encontraron valores nulos.")

st.divider()

# ─── SECCIÓN 2: Variables relevantes iniciales ──────────────────────────────
st.header("2. Variables relevantes — exploración inicial")

df_init = df.copy()
total_pedidos = len(df_init)
cancelados = (df_init["order_status"] == "cancelled").sum()
completados = total_pedidos - cancelados
tasa_general = cancelados / total_pedidos * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total pedidos", f"{total_pedidos:,}")
col2.metric("Completados", f"{completados:,}")
col3.metric("Cancelados", f"{cancelados:,}")
col4.metric("Tasa de cancelación", f"{tasa_general:.2f}%")

# Cancelación por ciudad
with st.expander("🏙️ Cancelaciones por ciudad"):
    cancelacion_ciudad = (
        df_init.groupby("restaurant_city")
        .agg(pedidos=("order_id", "count"),
             cancelados=("order_status", lambda x: (x == "cancelled").sum()))
        .assign(tasa_cancelacion=lambda x: x["cancelados"] / x["pedidos"] * 100)
        .sort_values("tasa_cancelacion", ascending=False)
    )
    st.dataframe(cancelacion_ciudad.round(2), use_container_width=True)

# Cancelación por tipo de cliente
with st.expander("👤 Cancelaciones y monto promedio por tipo de cliente"):
    cancelacion_cliente = (
        df_init.groupby("customer_loyalty_tier")
        .agg(pedidos=("order_id", "count"),
             cancelados=("order_status", lambda x: (x == "cancelled").sum()),
             monto_promedio=("total_amount", "mean"))
        .assign(tasa_cancelacion=lambda x: x["cancelados"] / x["pedidos"] * 100)
        .sort_values("tasa_cancelacion", ascending=False)
    )
    cancelacion_cliente["monto_promedio"] = cancelacion_cliente["monto_promedio"].round(2)
    cancelacion_cliente["tasa_cancelacion"] = cancelacion_cliente["tasa_cancelacion"].round(2).astype(str) + "%"
    st.dataframe(cancelacion_cliente, use_container_width=True)

st.divider()

# ─── SECCIÓN 3: Objetivo del proyecto ───────────────────────────────────────
st.header("3. Objetivo del proyecto")
st.info(
    "**Disminución y prevención de cancelaciones en pedidos de delivery** mediante el análisis de "
    "factores comerciales, operativos, climáticos y de comportamiento del cliente."
)
objetivos = [
    "Analizar el comportamiento de compra de los clientes.",
    "Evaluar el desempeño operativo de las entregas.",
    "Analizar el impacto en cancelaciones y promociones.",
    "Evaluar la relación entre las ciudades y tiempos de entrega según los diversos factores.",
    "Proponer estrategias de optimización del servicio de delivery.",
]
for i, obj in enumerate(objetivos, 1):
    st.markdown(f"**{i}.** {obj}")

st.divider()

# ─── Helpers ────────────────────────────────────────────────────────────────
df_val = df.copy()
df_val["is_cancelled"] = (df_val["order_status"] == "cancelled").astype(int)
df_val["delay_minutes"] = df_val["delivery_duration_actual"] - df_val["delivery_duration_estimated"]
df_val["tiene_descuento"] = df_val["discount_type"].notna()

def resumen_categoria(df_val, columna, tasa_general, total_pedidos):
    resumen = (
        df_val.groupby(columna, observed=False)
        .agg(pedidos=("order_id", "count"), cancelados=("is_cancelled", "sum"))
    )
    resumen["pct_pedidos"] = (resumen["pedidos"] / total_pedidos * 100).round(2)
    resumen["tasa_cancelacion"] = (resumen["cancelados"] / resumen["pedidos"] * 100).round(2)
    resumen["dif_vs_tasa_general"] = (resumen["tasa_cancelacion"] - tasa_general).round(2)
    return resumen.sort_values("tasa_cancelacion", ascending=False)

# ─── SECCIÓN 4: EDA orientado a objetivos ───────────────────────────────────
st.header("4. EDA orientado a los objetivos del proyecto")

st.markdown(
    """
Variables seleccionadas: `order_status`, `restaurant_city`, `delivery_duration_actual`,
`delivery_duration_estimated`, `customer_loyalty_tier`, `discount_type`, `total_amount`,
`is_weekend`, `is_holiday`, `weather_precipitation`.
"""
)

# ── 4.1 Pedidos completados vs cancelados ───────────────────────────────────
st.subheader("4.1 Pedidos completados vs cancelados")
status_counts = df["order_status"].value_counts()
total = status_counts.values.sum()

fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(
    status_counts.values,
    labels=status_counts.index,
    autopct=lambda pct: f"{int(round(pct * total / 100)):,} ({pct:.1f}%)",
    startangle=90,
)
ax.set_title("Proporción de pedidos completados y cancelados")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── 4.2 Categorías que INCIDEN en cancelación ───────────────────────────────
st.subheader("4.2 Categorías que inciden en la cancelación vs promedio")

# ── Ciudad ──────────────────────────────────────────────────────────────────
st.markdown(
    "### 🏙️ Ciudad\n"
    "Phoenix (9,2%) y Chicago (9,6%) tienden a cancelar por encima de la referencia general (7,8%)."
)

city_counts = df["restaurant_city"].value_counts().sort_values(ascending=True)
cancelacion_ciudad_eda = (
    df.groupby("restaurant_city")
    .agg(pedidos=("order_id", "count"),
         cancelados=("order_status", lambda x: (x == "cancelled").sum()))
    .assign(tasa_cancelacion=lambda x: x["cancelados"] / x["pedidos"] * 100)
    .sort_values("tasa_cancelacion", ascending=True)
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
plt.subplots_adjust(wspace=0.4)

colors1 = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(city_counts)))
bars1 = ax1.barh(city_counts.index, city_counts.values, color=colors1)
total_p = city_counts.sum()
for bar in bars1:
    w = bar.get_width()
    ax1.text(w + total_p * 0.005, bar.get_y() + bar.get_height() / 2,
             f"{int(w):,} ({w/total_p*100:.1f}%)", va="center")
ax1.set_xlim(0, city_counts.max() * 1.25)
ax1.set_xlabel("Cantidad de pedidos"); ax1.set_ylabel("Ciudad")
ax1.set_title("Cantidad de pedidos por ciudad")

colors2 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(cancelacion_ciudad_eda)))
bars2 = ax2.barh(cancelacion_ciudad_eda.index, cancelacion_ciudad_eda["tasa_cancelacion"], color=colors2)
for bar, (_, row) in zip(bars2, cancelacion_ciudad_eda.iterrows()):
    ax2.text(row["tasa_cancelacion"] + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{int(row['cancelados'])} ({row['tasa_cancelacion']:.1f}%)", va="center")
ax2.set_xlim(0, cancelacion_ciudad_eda["tasa_cancelacion"].max() * 1.25)
ax2.set_xlabel("Tasa de cancelación (%)"); ax2.set_ylabel("Ciudad")
ax2.set_title("Tasa de cancelación por ciudad")
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Delay ────────────────────────────────────────────────────────────────────
st.markdown(
    "### ⏱️ Delay en la entrega\n"
    "Las dos ciudades con mayor cancelación también muestran menor promedio de delay, "
    "sugiriendo mayor exigencia de puntualidad."
)

df_delay = df.copy()
df_delay["delay"] = df_delay["delivery_duration_actual"] - df_delay["delivery_duration_estimated"]
city_delay = (
    df_delay.groupby("restaurant_city")
    .agg(pedidos=("order_id", "count"), demora_promedio=("delay", "mean"))
    .sort_values("demora_promedio")
)

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(city_delay)))
bars = ax.barh(city_delay.index, city_delay["demora_promedio"], color=colors)
for bar, (_, row) in zip(bars, city_delay.iterrows()):
    ax.text(row["demora_promedio"] + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{row['demora_promedio']:.1f} min ({int(row['pedidos']):,} pedidos)",
            va="center", ha="left")
ax.axvline(0, color="black", linestyle="--", alpha=0.7)
ax.set_xlim(0, city_delay["demora_promedio"].max() + 2)
ax.set_xlabel("Demora promedio (Actual - Estimada)"); ax.set_ylabel("Ciudad")
ax.set_title("Demora promedio por ciudad")
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Precipitaciones ──────────────────────────────────────────────────────────
st.markdown(
    "### 🌧️ Precipitaciones\n"
    "Con lluvia, la cancelación baja ~1% respecto al promedio. "
    "Mayor empatía del cliente ante condiciones adversas."
)

df_rain = df[(df["weather_precipitation"] > 0) & (df["order_status"].isin(["completed", "cancelled"]))]
status_counts_rain = df_rain["order_status"].value_counts().reindex(["completed", "cancelled"]).fillna(0)
total_rain = status_counts_rain.sum()

fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(status_counts_rain,
       labels=["Completed", "Cancelled"],
       autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct * total_rain / 100)):,})",
       startangle=90)
ax.set_title(f"Órdenes con lluvia > 0: {total_rain:,.0f}")
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Categorías de comida ──────────────────────────────────────────────────────
st.markdown(
    "### 🍔 Categorías de comida\n"
    "Hamburguesas (8,6%) y Bebidas (8,4%) cancelan más; Menú ejecutivo (6,2%) y Postre (6,7%) menos."
)

item_counts = df["menu_item_category"].value_counts().sort_values(ascending=True)
cancelacion_item = (
    df.groupby("menu_item_category")
    .agg(pedidos=("order_id", "count"),
         cancelados=("order_status", lambda x: (x == "cancelled").sum()))
    .assign(tasa_cancelacion=lambda x: x["cancelados"] / x["pedidos"] * 100)
    .sort_values("tasa_cancelacion", ascending=True)
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
plt.subplots_adjust(wspace=0.4)
colors1 = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(item_counts)))
bars1 = ax1.barh(item_counts.index, item_counts.values, color=colors1)
total_p = item_counts.sum()
for bar in bars1:
    w = bar.get_width()
    ax1.text(w + total_p * 0.005, bar.get_y() + bar.get_height() / 2,
             f"{int(w):,} ({w/total_p*100:.1f}%)", va="center")
ax1.set_xlim(0, item_counts.max() * 1.25)
ax1.set_xlabel("Cantidad de pedidos"); ax1.set_ylabel("Categoría")
ax1.set_title("Cantidad de pedidos por categoría")
colors2 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(cancelacion_item)))
bars2 = ax2.barh(cancelacion_item.index, cancelacion_item["tasa_cancelacion"], color=colors2)
for bar, (_, row) in zip(bars2, cancelacion_item.iterrows()):
    ax2.text(row["tasa_cancelacion"] + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{int(row['cancelados'])} ({row['tasa_cancelacion']:.1f}%)", va="center")
ax2.set_xlim(0, cancelacion_item["tasa_cancelacion"].max() * 1.25)
ax2.set_xlabel("Tasa de cancelación (%)"); ax2.set_ylabel("Categoría")
ax2.set_title("Tasa de cancelación por Categoría")
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Canales de adquisición ────────────────────────────────────────────────────
st.markdown(
    "### 📣 Canales de adquisición de clientes\n"
    "`referral` (8,9%) y `paid_ads` (8,1%) muestran tasas más altas. "
    "Posible aprovechamiento de códigos sin intención real de compra."
)

channel = df["customer_acquisition_channel"].value_counts().sort_values(ascending=True)
cancelacion_canal = (
    df.groupby("customer_acquisition_channel")
    .agg(pedidos=("order_id", "count"),
         cancelados=("order_status", lambda x: (x == "cancelled").sum()))
    .assign(tasa_cancelacion=lambda x: x["cancelados"] / x["pedidos"] * 100)
    .sort_values("tasa_cancelacion", ascending=True)
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
plt.subplots_adjust(wspace=0.4)
colors1 = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(channel)))
bars1 = ax1.barh(channel.index, channel.values, color=colors1)
total_p = channel.sum()
for bar in bars1:
    w = bar.get_width()
    ax1.text(w + total_p * 0.005, bar.get_y() + bar.get_height() / 2,
             f"{int(w):,} ({w/total_p*100:.1f}%)", va="center")
ax1.set_xlim(0, channel.max() * 1.25)
ax1.set_xlabel("Cantidad de pedidos"); ax1.set_ylabel("Canal")
ax1.set_title("Cantidad de pedidos por Canal")
colors2 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(cancelacion_canal)))
bars2 = ax2.barh(cancelacion_canal.index, cancelacion_canal["tasa_cancelacion"], color=colors2)
for bar, (_, row) in zip(bars2, cancelacion_canal.iterrows()):
    ax2.text(row["tasa_cancelacion"] + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{int(row['cancelados'])} ({row['tasa_cancelacion']:.1f}%)", va="center")
ax2.set_xlim(0, cancelacion_canal["tasa_cancelacion"].max() * 1.25)
ax2.set_xlabel("Tasa de cancelación (%)"); ax2.set_ylabel("Canal")
ax2.set_title("Tasa de cancelación por Canal")
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Tipos de cocina ───────────────────────────────────────────────────────────
st.markdown(
    "### 🌍 Tipos de cocina\n"
    "Comida India y Mediterránea cancelan por arriba de la media. "
    "Difícil de explicar — punto para futuras exploraciones."
)

cuisine = df["restaurant_cuisine"].value_counts().sort_values(ascending=True)
cancelacion_cuisine = (
    df.groupby("restaurant_cuisine")
    .agg(pedidos=("order_id", "count"),
         cancelados=("order_status", lambda x: (x == "cancelled").sum()))
    .assign(tasa_cancelacion=lambda x: x["cancelados"] / x["pedidos"] * 100)
    .sort_values("tasa_cancelacion", ascending=True)
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
plt.subplots_adjust(wspace=0.4)
colors1 = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(cuisine)))
bars1 = ax1.barh(cuisine.index, cuisine.values, color=colors1)
total_p = cuisine.sum()
for bar in bars1:
    w = bar.get_width()
    ax1.text(w + total_p * 0.005, bar.get_y() + bar.get_height() / 2,
             f"{int(w):,} ({w/total_p*100:.1f}%)", va="center")
ax1.set_xlim(0, cuisine.max() * 1.25)
ax1.set_xlabel("Cantidad de pedidos"); ax1.set_ylabel("Categoría")
ax1.set_title("Cantidad de pedidos por categoría")
colors2 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(cancelacion_cuisine)))
bars2 = ax2.barh(cancelacion_cuisine.index, cancelacion_cuisine["tasa_cancelacion"], color=colors2)
for bar, (_, row) in zip(bars2, cancelacion_cuisine.iterrows()):
    ax2.text(row["tasa_cancelacion"] + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{int(row['cancelados'])} ({row['tasa_cancelacion']:.1f}%)", va="center")
ax2.set_xlim(0, cancelacion_cuisine["tasa_cancelacion"].max() * 1.25)
ax2.set_xlabel("Tasa de cancelación (%)"); ax2.set_ylabel("Categoría")
ax2.set_title("Tasa de cancelación por Categoría")
plt.tight_layout()
st.pyplot(fig); plt.close()

# ─── SECCIÓN 4.3: Categorías que NO inciden ─────────────────────────────────
st.subheader("4.3 Categorías que NO inciden en la cancelación vs promedio")

# ── Loyalty Tier ──────────────────────────────────────────────────────────────
st.markdown(
    "### 🥇 Loyalty Tier\n"
    "Los tiers Bronze, Silver y Gold presentan distribución homogénea — "
    "no explican diferencias en cancelaciones."
)

loyalty_counts = df["customer_loyalty_tier"].value_counts().sort_values(ascending=True)
cancelacion_loyalty = (
    df.groupby("customer_loyalty_tier")
    .agg(pedidos=("order_id", "count"),
         cancelados=("order_status", lambda x: (x == "cancelled").sum()))
    .assign(tasa_cancelacion=lambda x: x["cancelados"] / x["pedidos"] * 100)
    .sort_values("tasa_cancelacion", ascending=True)
)
color_map = {"Bronze": "#cd7f32", "Silver": "#c0c0c0", "Gold": "#ffd700"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
plt.subplots_adjust(wspace=0.4)
bars1 = ax1.barh(loyalty_counts.index, loyalty_counts.values,
                  color=[color_map.get(i, "grey") for i in loyalty_counts.index])
total_p = loyalty_counts.sum()
for bar in bars1:
    w = bar.get_width()
    ax1.text(w + total_p * 0.01, bar.get_y() + bar.get_height() / 2,
             f"{int(w):,} ({w/total_p*100:.1f}%)", va="center")
ax1.set_xlim(0, loyalty_counts.max() * 1.2)
ax1.set_xlabel("Cantidad de pedidos"); ax1.set_ylabel("Tipo de cliente")
ax1.set_title("Cantidad de pedidos por tipo de cliente")
bars2 = ax2.barh(cancelacion_loyalty.index, cancelacion_loyalty["tasa_cancelacion"],
                  color=[color_map.get(i, "grey") for i in cancelacion_loyalty.index])
for bar, (_, row) in zip(bars2, cancelacion_loyalty.iterrows()):
    ax2.text(row["tasa_cancelacion"] + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{int(row['cancelados'])} ({row['tasa_cancelacion']:.1f}%)", va="center")
ax2.set_xlim(0, cancelacion_loyalty["tasa_cancelacion"].max() * 1.2)
ax2.set_xlabel("Tasa de cancelación (%)"); ax2.set_ylabel("Tipo de cliente")
ax2.set_title("Tasa de cancelación por tipo de cliente")
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Temperatura ───────────────────────────────────────────────────────────────
st.markdown(
    "### 🌡️ Temperatura\n"
    "La temperatura promedio es muy similar entre pedidos completados y cancelados — no incide."
)

status_stats = (
    df.groupby("order_status")
    .agg(pedidos=("order_id", "count"), temperatura_promedio=("weather_temperature", "mean"))
    .sort_values("order_status", ascending=True)
)

fig, ax = plt.subplots(figsize=(12, 4))
bars = ax.barh(status_stats.index, status_stats["temperatura_promedio"])
for bar, (_, row) in zip(bars, status_stats.iterrows()):
    ax.text(row["temperatura_promedio"] + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{row['temperatura_promedio']:.1f}°C ({int(row['pedidos']):,} pedidos)", va="center")
ax.set_xlim(0, status_stats["temperatura_promedio"].max() * 1.2)
ax.set_xlabel("Temperatura promedio (°C)"); ax.set_ylabel("Estado del pedido")
ax.set_title("Temperatura promedio por estado del pedido")
plt.tight_layout()
st.pyplot(fig); plt.close()

# Temperatura por rangos
df_temp = df.copy()
df_temp["temp_categoria"], bins_t = pd.qcut(
    df_temp["weather_temperature"], q=3,
    labels=["Baja", "Media", "Alta"], retbins=True)
temp_labels = {
    "Baja":  f"Baja ({bins_t[0]:.1f}° a {bins_t[1]:.1f}°)",
    "Media": f"Media ({bins_t[1]:.1f}° a {bins_t[2]:.1f}°)",
    "Alta":  f"Alta ({bins_t[2]:.1f}° a {bins_t[3]:.1f}°)",
}
temp_status = pd.crosstab(df_temp["order_status"], df_temp["temp_categoria"])
temp_status_pct = temp_status.div(temp_status.sum(axis=1), axis=0) * 100
temp_status_pct.columns = [temp_labels[c] for c in temp_status_pct.columns]
temp_status.columns = [temp_labels[c] for c in temp_status.columns]

fig, ax = plt.subplots(figsize=(12, 6))
temp_status_pct.plot(kind="bar", stacked=True, ax=ax, colormap="coolwarm")
for ci, container in enumerate(ax.containers):
    counts = temp_status.iloc[:, ci].values
    labels = [
        f"{c}\n({p:.1f}%)" if p >= 5 else ""
        for c, p in zip(counts, container.datavalues)
    ]
    ax.bar_label(container, labels=labels, label_type="center", fontsize=9)
ax.set_title("Distribución de rangos de temperatura por estado del pedido")
ax.set_xlabel("Estado del pedido"); ax.set_ylabel("Porcentaje (%)")
ax.legend(title="Rango de temperatura", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Descuentos ────────────────────────────────────────────────────────────────
st.markdown(
    "### 🏷️ Descuentos\n"
    "Los tipos de descuento tampoco resultan relevantes — distribución homogénea."
)

discount = df["discount_type"].value_counts().sort_values(ascending=True)
cancelacion_discount = (
    df.groupby("discount_type")
    .agg(pedidos=("order_id", "count"),
         cancelados=("order_status", lambda x: (x == "cancelled").sum()))
    .assign(tasa_cancelacion=lambda x: x["cancelados"] / x["pedidos"] * 100)
    .sort_values("tasa_cancelacion", ascending=True)
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
plt.subplots_adjust(wspace=0.4)
colors1 = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(discount)))
bars1 = ax1.barh(discount.index, discount.values, color=colors1)
total_p = discount.sum()
for bar in bars1:
    w = bar.get_width()
    ax1.text(w + total_p * 0.005, bar.get_y() + bar.get_height() / 2,
             f"{int(w):,} ({w/total_p*100:.1f}%)", va="center")
ax1.set_xlim(0, discount.max() * 1.25)
ax1.set_xlabel("Cantidad de pedidos"); ax1.set_ylabel("Tipo de descuento")
ax1.set_title("Cantidad de pedidos por Tipo de descuento")
colors2 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(cancelacion_discount)))
bars2 = ax2.barh(cancelacion_discount.index, cancelacion_discount["tasa_cancelacion"], color=colors2)
for bar, (_, row) in zip(bars2, cancelacion_discount.iterrows()):
    ax2.text(row["tasa_cancelacion"] + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{int(row['cancelados'])} ({row['tasa_cancelacion']:.1f}%)", va="center")
ax2.set_xlim(0, cancelacion_discount["tasa_cancelacion"].max() * 1.25)
ax2.set_xlabel("Tasa de cancelación (%)"); ax2.set_ylabel("Tipo de descuento")
ax2.set_title("Tasa de cancelación por Tipo de descuento")
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Feriados y fines de semana ────────────────────────────────────────────────
st.markdown(
    "### 📅 Días de la semana y días especiales\n"
    "La cancelación se mantiene estable — ni feriados ni fines de semana impactan."
)

for col_bool, labels_map, title_suffix in [
    ("is_holiday", {True: "Feriado", False: "No feriado"}, "feriados vs no feriados"),
    ("is_weekend", {True: "Fin de semana", False: "Día de semana"}, "distribuidos en la semana"),
]:
    serie = df[col_bool].map(labels_map).value_counts().sort_values(ascending=True)
    cancel_ser = (
        df.assign(**{col_bool: df[col_bool].map(labels_map)})
        .groupby(col_bool)
        .agg(pedidos=("order_id", "count"),
             cancelados=("order_status", lambda x: (x == "cancelled").sum()))
        .assign(tasa_cancelacion=lambda x: x["cancelados"] / x["pedidos"] * 100)
        .reindex(serie.index)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 3))
    plt.subplots_adjust(wspace=0.4)
    colors1 = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(serie)))
    bars1 = ax1.barh(serie.index, serie.values, color=colors1)
    total_p = serie.sum()
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + total_p * 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{int(w):,} ({w/total_p*100:.1f}%)", va="center")
    ax1.set_xlim(0, serie.max() * 1.25)
    ax1.set_xlabel("Cantidad de pedidos")
    ax1.set_title(f"Cantidad de pedidos {title_suffix}")
    colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(cancel_ser)))
    bars2 = ax2.barh(cancel_ser.index, cancel_ser["tasa_cancelacion"], color=colors2)
    for bar, (_, row) in zip(bars2, cancel_ser.iterrows()):
        ax2.text(row["tasa_cancelacion"] + 0.2, bar.get_y() + bar.get_height() / 2,
                 f"{int(row['cancelados'])} ({row['tasa_cancelacion']:.1f}%)", va="center")
    ax2.set_xlim(0, cancel_ser["tasa_cancelacion"].max() * 1.25)
    ax2.set_xlabel("Tasa de cancelación (%)")
    ax2.set_title(f"Tasa de cancelación {title_suffix}")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.divider()

# ─── SECCIÓN 5: Modelo predictivo ───────────────────────────────────────────
st.header("5. Modelo predictivo para cancelaciones")

st.markdown(
    """
**¿Por qué Regresión Logística?**
- Es **interpretable**: muestra cuánto contribuye cada variable.
- Maneja variables **numéricas y categóricas** combinadas.
- Es un **buen punto de partida** antes de modelos más complejos como XGBoost.
"""
)

if st.button("🚀 Entrenar modelo", type="primary"):
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    with st.spinner("Entrenando modelo…"):
        df_model = df.copy()
        df_model["cancelled"] = (df_model["order_status"] == "cancelled").astype(int)

        features = [
            "restaurant_city", "restaurant_cuisine", "menu_item_category",
            "customer_acquisition_channel", "discount_type",
            "weather_temperature", "weather_precipitation",
            "delivery_duration_estimated", "is_weekend", "is_holiday",
        ]
        target = "cancelled"
        X = df_model[features]
        y = df_model[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y)

        categorical_features = [
            "restaurant_city", "restaurant_cuisine", "menu_item_category",
            "customer_acquisition_channel", "discount_type"]
        numeric_features = [
            "weather_temperature", "weather_precipitation",
            "delivery_duration_estimated", "is_weekend", "is_holiday"]

        preprocessor = ColumnTransformer(transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_features),
        ])

        logistic_model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ])
        logistic_model.fit(X_train, y_train)
        y_pred = logistic_model.predict(X_test)
        y_prob = logistic_model.predict_proba(X_test)[:, 1]

    st.success("✅ Modelo entrenado")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("ROC AUC", f"{roc_auc_score(y_test, y_prob):.4f}")

    with col2:
        cm = confusion_matrix(y_test, y_pred)
        st.write("**Confusion Matrix**")
        st.dataframe(pd.DataFrame(
            cm,
            index=["Real: completado", "Real: cancelado"],
            columns=["Pred: completado", "Pred: cancelado"],
        ))

    st.write("**Classification Report**")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    feature_names = logistic_model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pd.DataFrame({
        "feature": feature_names,
        "coefficient": logistic_model.named_steps["classifier"].coef_[0],
    })

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top features que AUMENTAN cancelación**")
        st.dataframe(
            coefficients.sort_values("coefficient", ascending=False).head(20).reset_index(drop=True),
            use_container_width=True)
    with col2:
        st.write("**Top features que REDUCEN cancelación**")
        st.dataframe(
            coefficients.sort_values("coefficient", ascending=True).head(20).reset_index(drop=True),
            use_container_width=True)

st.divider()

# ─── SECCIÓN 6: Conclusiones ─────────────────────────────────────────────────
st.header("6. Conclusiones y recomendaciones")

st.markdown(
    """
A partir del análisis realizado, la tasa general de cancelación es de aproximadamente **7,8%**,
usada como referencia comparativa.

**Lo que no explica las cancelaciones:**
- Nivel de lealtad (Bronze, Silver, Gold) — distribución homogénea.
- Temperatura — promedios muy similares en completados y cancelados.
- Feriados y fines de semana — cancelación estable.
- Tipos de descuento — sin diferencias significativas.

**Lo que SÍ incide:**
- **Ciudad:** Phoenix y Chicago superan el promedio. Chicago es clave por su alto volumen.
  San Diego es la ciudad modelo (alto volumen, baja cancelación).
- **Categoría de comida:** Hamburguesas y Bebidas cancelan más; platos elaborados menos.
  Clientes con pedidos complejos muestran mayor tolerancia.
- **Canal de adquisición:** `referral` y `paid_ads` con tasas más altas.
  Posible aprovechamiento de códigos sin intención real de compra.
- **Delay:** Las ciudades con mayor cancelación también tienen menor delay promedio,
  sugiriendo exigencia de puntualidad mayor.
- **Precipitaciones:** Con lluvia, la cancelación baja ~1% — mayor empatía del cliente.

**Recomendaciones:**
1. Focalizar acciones en **Chicago y Phoenix** — mejor comunicación ante demoras y seguimiento del pedido.
2. Revisar la estrategia de **referrals y paid ads** para asegurar captación de usuarios con intención real.
3. Ampliar las estimaciones de tiempo de entrega en ciudades con alta cancelación.
4. Considerar incentivos específicos para categorías de alto abandono (hamburguesas, bebidas).

> *Este análisis se realizó sobre un dataset sintético. Las conclusiones deben interpretarse
> como hipótesis analíticas, no como causalidades definitivas.*
"""
)
