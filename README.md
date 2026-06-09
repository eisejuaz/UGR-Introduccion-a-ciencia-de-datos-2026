# Predicción y Análisis de Cancelaciones en Pedidos de Delivery

Aplicación interactiva desarrollada con Streamlit para analizar los factores que influyen en la cancelación de pedidos de delivery y construir un modelo predictivo capaz de identificar órdenes con mayor probabilidad de cancelación.

\---

## Objetivo del Proyecto

Las cancelaciones representan una pérdida operativa y económica para las plataformas de delivery.

El objetivo principal es:

* Identificar variables asociadas a las cancelaciones.
* Analizar patrones de comportamiento de clientes y restaurantes.
* Detectar oportunidades de mejora operativa.
* Construir un modelo predictivo para anticipar cancelaciones.
* Generar recomendaciones de negocio basadas en datos.

\---

## Preguntas de Negocio

El análisis busca responder:

* ¿Qué ciudades presentan mayores tasas de cancelación?
* ¿Influyen las demoras en las cancelaciones?
* ¿Las condiciones climáticas afectan el comportamiento del cliente?
* ¿Existen categorías de comida más propensas a ser canceladas?
* ¿Qué canales de adquisición generan clientes de menor calidad?
* ¿Es posible predecir una cancelación antes de que ocurra?

\---

## Funcionalidades de la Aplicación

### Exploración Inicial del Dataset

* Vista previa del dataset
* Tipos de datos
* Estadísticas descriptivas
* Detección de valores nulos
* Análisis de duplicados
* Cardinalidad de variables

### Indicadores Iniciales

Visualización de KPIs:

* Total de pedidos
* Pedidos completados
* Pedidos cancelados
* Tasa general de cancelación

Además:

* Cancelación por ciudad
* Cancelación por tipo de cliente
* Ticket promedio por segmento

### Definición del Problema

Presentación de los objetivos analíticos y contexto de negocio.

### Análisis Exploratorio (EDA)

Se estudian variables relacionadas con:

#### Operación

* Tiempo estimado de entrega
* Tiempo real de entrega
* Delay de entrega

#### Cliente

* Loyalty Tier
* Canal de adquisición

#### Producto

* Categoría del menú
* Tipo de cocina

#### Contexto

* Ciudad
* Temperatura
* Precipitaciones
* Feriados
* Fin de semana

### Modelo Predictivo

Se implementa un modelo de **Regresión Logística**.

Características:

* One-Hot Encoding para variables categóricas.
* Imputación automática de valores faltantes.
* División Train/Test.
* Balanceo de clases.
* Evaluación mediante métricas de clasificación.

Métricas mostradas:

* ROC AUC
* Classification Report
* Confusion Matrix

También se visualizan:

* Variables que aumentan la probabilidad de cancelación.
* Variables que reducen la probabilidad de cancelación.

### Conclusiones y Recomendaciones

Se presentan hallazgos accionables para:

* Reducir cancelaciones.
* Mejorar la experiencia del cliente.
* Optimizar tiempos de entrega.
* Mejorar campañas de adquisición.

\---

## Tecnologías Utilizadas

|Tecnología|Uso|
|-|-|
|Python|Desarrollo principal|
|Streamlit|Aplicación web|
|Pandas|Manipulación de datos|
|NumPy|Cálculo numérico|
|Matplotlib|Visualizaciones|
|Scikit-Learn|Machine Learning|
|GDown|Descarga automática del dataset|

\---

## Estructura del Proyecto

```text
.
├── app.py
├── requirements.txt
├── README.md
└── dataset
```

\---

## Ejecución Local

#### 1\. Clonar repositorio

```bash
git clone https://github.com/usuario/repositorio.git
cd repositorio
```

#### 2\. Instalar dependencias

```bash
pip install -r requirements.txt
```

#### 3\. Ejecutar aplicación

```bash
streamlit run app.py
```

\---

## Despliegue en Streamlit Cloud

1. Subir el proyecto a GitHub.
2. Crear una nueva aplicación en Streamlit Cloud.
3. Seleccionar el repositorio y `app.py` como archivo principal.
4. Realizar el deploy.

\---

## Principales Hallazgos

* Chicago y Phoenix presentan tasas de cancelación superiores al promedio.
* Los pedidos de hamburguesas y bebidas muestran mayor propensión a cancelarse.
* Los clientes adquiridos por Referral y Paid Ads presentan mayor riesgo.
* La lluvia no incrementa las cancelaciones.
* El nivel de fidelización no presenta diferencias significativas.
* Las demoras percibidas parecen tener mayor impacto que las demoras reales.

\---

## Mejoras Futuras

* Implementar XGBoost y Random Forest.
* Agregar SHAP para interpretabilidad.
* Incorporar predicción en tiempo real.
* Crear dashboard ejecutivo.
* Desarrollar sistema de alertas preventivas.
* Integrar bases de datos productivas.

\---

## Autores

* Julian Deheza
* Franco Gargiulo
* Mauro Girondo
* Melisa Tusoli

**Tecnicatura Universitaria en Ciencia de Datos e Inteligencia Artificial – 2026**

