# Dashboard de Análisis de MLP con MNIST

Este proyecto es una aplicación interactiva desarrollada en **Streamlit** para explorar, analizar y visualizar el rendimiento de redes neuronales Perceptrón Multicapa (MLP) entrenadas sobre el dataset MNIST (dígitos escritos a mano).

## 📋 Descripción

La aplicación guía al usuario a través de varios ejercicios prácticos de Deep Learning:
1. **Exploración de Hiperparámetros**: Permite seleccionar modelos basados en resultados previos de una búsqueda en grilla (Grid Search), variando el número de capas, neuronas, optimizadores, épocas y tamaño de lote.
2. **Comparación de Arquitecturas**: Visualiza y compara la definición de modelos usando la API Secuencial vs. la API Funcional de Keras.
3. **Análisis de Desempeño**:
   - Generación de reportes de clasificación y matrices de confusión.
   - Visualización de curvas de Precisión-Recall por clase.
   - Inspección visual de errores (imágenes mal clasificadas).

## 🛠️ Requisitos e Instalación

Se recomienda utilizar Python 3.8 o superior. Instala las dependencias necesarias ejecutando:

```bash
pip install streamlit tensorflow pandas numpy matplotlib seaborn scikit-learn tf-keras
```

## 🚀 Ejecución

Sitúate en el directorio raíz del proyecto (`/workspaces/MLP/`) y ejecuta:

```bash
streamlit run streamlit/app.py
```

O ir a [Streamlit Cloud](https://mpldeep.streamlit.app/)

## 🧠 Funcionalidades Detalladas

### Ejercicio 1: Optimización
Selecciona una configuración de modelo en el panel lateral o principal. La app intentará cargar los pesos desde las carpetas `GS_...` correspondientes usando el `trial_id` encontrado en los archivos de configuración.

### Ejercicio 2: Arquitecturas
Muestra ejemplos de código y resúmenes (`summary`) de modelos construidos con:
- **Sequential API**: Estructura lineal simple.
- **Functional API**: Para topologías más complejas y flexibles.

### Ejercicio 4: Interpretación
Genera visualizaciones en tiempo real sobre el conjunto de test de MNIST:
- **Matriz de Confusión**: Identifica qué dígitos se confunden entre sí (ej. 4 con 9).
- **Curvas P-R**: Encuentra el umbral óptimo de decisión.
- **Galería de Errores**: Muestra las imágenes reales que el modelo predijo incorrectamente para entender las fallas.
