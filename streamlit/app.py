import streamlit as st
import tensorflow as tf
import pandas as pd
import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np  # Biblioteca para operaciones matemáticas y manejo de matrices
import matplotlib.pyplot as plt  # Para graficar datos y visualizar resultados
from sklearn.metrics import precision_recall_curve

# Importa el conjunto de datos MNIST, un dataset clásico para clasificación de dígitos escritos a mano
from tf_keras.datasets import mnist
from tf_keras import models, layers
from tf_keras.utils import to_categorical

### DATOS
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 784))
test_images = test_images.reshape((10000, 784))

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_labels_cod = to_categorical(train_labels)
test_labels_cod = to_categorical(test_labels)

st.set_page_config(layout="wide")

st.markdown(
    """
    # Ejercicio 1: Optimización de Hiperparámetros para Máxima Precisión

    ## 📌 Objetivo
    El objetivo de este ejercicio es encontrar la mejor combinación de hiperparámetros para maximizar la precisión en el conjunto de prueba.
    
    ---
    ## 🔧 Hiperparámetros a experimentar
    
    - **Optimizador**: `adam`, `sgd`.
    - **Tamaño de batch**: `32`, `64`, `128`.
    - **Número de épocas**: `10`, `20`.
    - **Arquitectura de la red**: Variación en el número de neuronas y capas densas.
    """ 
)

st.code("""
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(train_images.shape[1],)))  # Verifica que el tamaño sea correcto
    
    # Número de capas densas
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(layers.Dense(
            units=hp.Choice(f'capa{i}_neuronas', [32, 64, 128]),
            activation='relu'
        ))
    
    model.add(layers.Dense(10, activation='softmax'))  # Ajusta según el número de clases
    
    model.compile(
        optimizer=hp.Choice('optimizer', ['adam', 'sgd']),
        loss=tf.keras.losses.CategoricalCrossentropy(),  # Corregido
        metrics=['accuracy', tf.keras.metrics.F1Score(average='micro')]  # Corregido
    )
    
    return model

# Definir el tuner para la búsqueda en grilla
tuner = kt.GridSearch(
    build_model,
    objective='val_accuracy',
    executions_per_trial=1,
)

hp = kt.HyperParameters()
tuner.search(train_images, to_categorical(train_labels), 
             epochs=hp.Choice('epochs', [10, 20]),
             batch_size=hp.Choice('batch_size', [32, 64, 128]),
             validation_data=(test_images, to_categorical(test_labels)))

        
>>> Trial 78 Complete [00h 00m 34s]
>>> val_accuracy: 0.9692000150680542

>>> Best val_accuracy So Far: 0.9818999767303467
>>> Total elapsed time: 02h 33m 06s

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

>>> Mejores hiperparámetros: {'num_layers': 3, 'capa0_neuronas': 128, 'optimizer': 'adam', 'capa1_neuronas': 128, 'capa2_neuronas': 128}
""")

st.markdown(
"""
## Todos los modelos
""")

df = pd.read_csv("streamlit/output.csv")#.drop(columns="Unnamed: 0")

col1, col2 = st.columns([0.7,0.3])

with col1:
    st.dataframe(df,height=500,hide_index=True) #.style.highlight_max(axis=0, subset="test_f1", color='lightgreen')}

with col2:
    st.markdown("### Selecione un modelo")
    capas = st.selectbox("Seleccione numero de capas:",[1,2,3])

    numcols = [1, 0.001, 0.001] if capas == 1 else [0.5, 0.5, 0.001] if capas == 2 else [0.3, 0.3, 0.3]


    col1, col2, col3 = st.columns(numcols)
    
    with col1:
        neuronas_capa1 = st.selectbox("Seleccione numero de neuronas capa 1:",[32,64,128])
    
    with col2:
        neuronas_capa2 = st.selectbox("Seleccione numero de neuronas capa 2:",[32,64,128]) if capas >= 2 else None
    
    with col3:
        neuronas_capa3 = st.selectbox("Seleccione numero de neuronas capa 3:",[32,64,128]) if capas == 3 else None

    optimizador = st.selectbox("Seleccione optimizador:",["adam","sgd"])
    epocas = st.selectbox("Seleccione numero de epocas:",[10,20])
    lotes = st.selectbox("Seleccione tamaño de lote:",[32, 64, 128])

hps = {'num_layers': capas, 'capa0_neuronas': neuronas_capa1, 'optimizer': optimizador} if capas == 1 else \
      {'num_layers': capas, 'capa0_neuronas': neuronas_capa1, 'optimizer': optimizador, 'capa1_neuronas': neuronas_capa2} if capas == 2 else \
      {'num_layers': capas, 'capa0_neuronas': neuronas_capa1, 'optimizer': optimizador, 'capa1_neuronas': neuronas_capa2, 'capa2_neuronas': neuronas_capa3}


folder_path = f"GS_E{epocas}BS_{lotes}"

#folder_path = os.path.join(mlp_path, folder_name)

for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                for file_name in os.listdir(dir_path):
                    if file_name.startswith("trial"):
                        trial = os.path.join(dir_path, file_name)
                        with open(trial, "r") as f:
                            config = json.load(f)

                        conf_hps = config["hyperparameters"]["values"]
    
                        if conf_hps == hps:
                            trial_id = config["trial_id"]
                             
num_layers = hps["num_layers"]
neurons_per_layer = [hps[f"capa{i}_neuronas"] for i in range(num_layers)]
optimizer = hps["optimizer"]

# Construir el modelo
model = Sequential()
model.add(Dense(neurons_per_layer[0], activation="relu", input_shape=(train_images.shape[1],)))  # Ajusta input_dim según tu dataset

for neurons in neurons_per_layer[1:]:
    model.add(Dense(neurons, activation="relu"))

model.add(Dense(10, activation="sigmoid"))  # Asumiendo una salida binaria, cambia según tu caso

# Compilar el modelo
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Cargar pesos
model.load_weights(f"GS_E{epocas}BS_{lotes}/trial_{trial_id}/checkpoint.weights.h5")

import io

buffer = io.StringIO()
model.summary(print_fn=lambda x: buffer.write(x + "\n"))
summary_string = buffer.getvalue()
buffer.close()


st.markdown("""
---
# Ejercicio 2: Flexibilidad Avanzada en la Definición del modelo

### Arquitectura de MMM: Tres Enfoques
Exploramos tres maneras de definir un modelo de aprendizaje profundo: **Secuencial**, **Funcional** y **Subclassing**.

### Comparación de Enfoques
- **Claridad y rapidez**: Cada método tiene sus ventajas según la complejidad del modelo.
- **Flexibilidad**: Algunos enfoques permiten mayor modularidad y personalización.

### Visualización de la Estructura
Se utilizará `plot_model` para representar gráficamente cada arquitectura.

### Extra: Normalización de Lotes
Se añadirá `BatchNormalization` después de cada capa oculta para analizar su impacto en la estructura del modelo.

---
""")

st.markdown("""
## Arquitectura Secuencial
""")

model_sequential = models.Sequential()

model_sequential.add(layers.Dense(128, activation='relu', input_shape=(train_images.shape[1],)))
model_sequential.add(layers.BatchNormalization())
model_sequential.add(layers.Dense(128, activation='relu'))
model_sequential.add(layers.BatchNormalization())
model_sequential.add(layers.Dense(128, activation='relu'))
model_sequential.add(layers.BatchNormalization())
model_sequential.add(layers.Dense(10, activation='sigmoid'))

model_sequential.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.F1Score(average='micro')])

col1, col2 = st.columns(2)

with col1:
    st.code("""
    model_sequential = models.Sequential()

    model_sequential.add(
            layers.Dense(128, 
                        activation='relu', 
                        input_shape=(train_images.shape[1],)))
            
    model_sequential.add(
            layers.BatchNormalization()
            )
    
    model_sequential.add(
            layers.Dense(128, activation='relu')
            )
    
    model_sequential.add(
            layers.BatchNormalization()
            )
    
    model_sequential.add(
            layers.Dense(128, activation='relu')
            )
    
    model_sequential.add(
            layers.BatchNormalization()
            )
    
    model_sequential.add(
            layers.Dense(10, activation='sigmoid')
            )

    model_sequential.compile(optimizer="adam", 
                             loss='binary_crossentropy', }
                             metrics=['accuracy',tf.keras.metrics.F1Score(average='micro')])
    """)

buffer = io.StringIO()
model_sequential.summary(print_fn=lambda x: buffer.write(x + "\n"))
summary_string = buffer.getvalue()
buffer.close()

with col2:
    st.code(summary_string, language="text")


st.markdown("""
---

## Arquitectura Funcional   
""")

import tensorflow as tf
from tensorflow.keras import layers

inputs = tf.keras.Input(shape=(train_images.shape[1],))  

x = layers.Dense(128, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)

outputs = layers.Dense(10, activation='sigmoid')(x)

model_funtional = tf.keras.Model(inputs=inputs, outputs=outputs)

model_funtional.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.F1Score(average='micro')])
model_funtional.summary()

col1, col2 = st.columns([0.4,0.6])

with col1:
    st.code("""
inputs = tf.keras.Input(shape=(train_images.shape[1],))  

x = layers.Dense(128, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)

outputs = layers.Dense(10, activation='sigmoid')(x)

model_funtional = tf.keras.Model(inputs=inputs, 
            outputs=outputs)

model_funtional.compile(
            optimizer="adam", 
            loss='binary_crossentropy', 
            metrics=['accuracy',
                     tf.keras.metrics.F1Score(
                        average='micro'
                        )
                    ]
            )
""")

buffer = io.StringIO()
model_funtional.summary(print_fn=lambda x: buffer.write(x + "\n"))
summary_string = buffer.getvalue()
buffer.close()

with col2:
    st.code(summary_string, language="text")


st.markdown("""
---

# Ejercicio 3: Explorando y Extendiendo `evaluate` y `predict`

1. Investiga las opciones avanzadas de las funciones `evaluate` y `predict` en TensorFlow/Keras. Realiza un ejemplo de la utilidad de cada una.

2. Investiga cómo cambian las métricas de desempeño de MMM cuando cambia el tamaño de lote.
""")

st.code("""
Model.predict(
    
    x
    - Un numpy array o una lista de arrays.
    - Un tensor o una lista de tensores.
    - Un tf.data.Dataset, útil para manejar grandes volúmenes de datos de manera eficiente.
    - Una objeto de keras.utils.PyDataset.

    batch_size Especifica cuántas muestras se procesan en cada iteración.
    Si es None, el valor predeterminado es 32.
    No debes especificarlo si x es un tf.data.Dataset, un generador o un keras.utils.PyDataset, ya que estos ya manejan los lotes automáticamente.

    verbose (Nivel de detalle en la salida)
    - "auto": Se comporta como 1 en la mayoría de los casos.
    - 0: No muestra nada.
    - 1: Muestra una barra de progreso.
    - 2: Muestra una línea de salida por cada iteración, útil cuando se ejecuta en producción o se guarda en un archivo de logs.

    steps** (Número total de pasos):
    - Si es None, se procesa hasta agotar los datos.
    - Si x es un tf.data.Dataset, y steps=None, evaluará hasta que los datos se terminen.
    - si steps=N, se ejecutarán N lotes y luego la predicción se detendrá, sin importar si hay más datos en el dataset.

    callbacks (Lista de funciones de callback)
    Los callbacks son objetos que permiten ejecutar funciones personalizadas mientras el modelo está haciendo predicciones.
)

model.evaluate(
    X
    - Un NumPy array o lista de NumPy arrays (si el modelo tiene múltiples entradas).
    - Un tensor o lista de tensores.
    - Un diccionario donde las claves son los nombres de las entradas del modelo y los valores son los datos correspondientes.
    - Un tf.data.Dataset, que debe devolver (inputs, targets) o (inputs, targets, sample_weights).
    - Un generador o keras.utils.PyDataset que devuelva (inputs, targets) o (inputs, targets, sample_weights).

    y
    Son las etiquetas o valores reales que se compararán con las predicciones del modelo.
    Deben ser un NumPy array o un tensor, igual que x.
    Si x es un tf.data.Dataset o un keras.utils.PyDataset, no se debe especificar y, ya que los datos de salida se obtienen automáticamente del dataset.

    batch_size (tamaño del lote)
    Número de muestras procesadas en cada paso.
    Si no se especifica, se usa 32 por defecto.
    No usar este parámetro si x es un tf.data.Dataset, generador o keras.utils.PyDataset, ya que estos ya manejan los lotes automáticamente.

    verbose
    "auto": Modo automático (1 en la mayoría de los casos).
    0: No muestra nada (modo silencioso).
    1: Muestra una barra de progreso.
    2: Muestra solo una línea por cada paso.

    sample_weight
    Un NumPy array opcional para dar más importancia a ciertas muestras al calcular la pérdida.
    Puede ser:
    Un array 1D con el mismo número de elementos que x (cada muestra tiene un peso).
    Un array 2D con forma (muestras, longitud_secuencia), útil para datos secuenciales donde cada paso en la secuencia tiene un peso diferente.

    steps
    Número total de lotes (batches) que se ejecutarán antes de terminar la evaluación.
    Si es None, se evaluarán todos los datos.
    Si x es un tf.data.Dataset y steps=None, la evaluación continuará hasta que el dataset se agote.

    callbacks
    Lista de instancias de keras.callbacks.Callback para ejecutar acciones durante la evaluación.
    Útil para monitorear métricas o guardar información en tiempo real.

    return_dict
    False (por defecto): Devuelve la pérdida y las métricas como una lista de valores.
    True: Devuelve un diccionario donde las claves son los nombres de las métricas y los valores sus respectivos resultados.            
)
""")

st.markdown("""
# Ejercicio 4: Interpretación y Análisis del Desempeño

1.
   - **Precisión por Clase**:
     - Calcula y interpreta el desempeño del modelo para cada clase.
   - **Matriz de Confusión**:
     - Genera una matriz de confusión para identificar las clases más confundidas.
   - **Confianza en las Predicciones**:
     - Crea histogramas para visualizar la distribución de probabilidades predichas por clase.
   - **Análisis de Errores**:
     - Selecciona ejemplos mal clasificados, visualiza las imágenes y discute posibles razones de los errores.
   - Encuentre el umbral de mejor desempeño en cada clase.

---
""")

st.markdown(
"""
## Desempeño del modelo seleccionado:
""")

import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 🚀 Hacer predicciones con el modelo
y_pred_probs = model.predict(test_images)  # Probabilidades por clase
y_pred_classes = np.argmax(y_pred_probs, axis=1)  # Clases predichas

report = classification_report(test_labels, y_pred_classes, digits=4)

# Capturar la salida en un StringIO
buffer = io.StringIO()
print(report, file=buffer)
classification_str = buffer.getvalue()
buffer.close()

col1, col2 = st.columns([0.4,0.6])

with col1:
    st.text("Reporte de clasificación:")
    st.code(classification_str, language="text")

# 2️⃣ **Matriz de Confusión**
cm = confusion_matrix(test_labels, y_pred_classes)
plt.figure(figsize=(8, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(len(cm)), yticklabels=range(len(cm)))
plt.xlabel("Predicción")
plt.ylabel("Clase real")
plt.title("Matriz de Confusión")
with col2:
    st.pyplot(plt)

#print("\n🔹 Mejor umbral y F1-score para cada clase:")
#for cls, (thresh, f1) in best_thresholds.items():
#    print(f"Clase {cls}: Umbral = {thresh:.4f}, F1-score = {f1:.4f}")

digito = st.selectbox("Eliga un digito:",sorted(np.unique(test_labels)))

precision, recall, thresholds = precision_recall_curve(to_categorical(test_labels)[:, digito], y_pred_probs[:, digito])
f1_scores = 2 * (precision * recall) / (precision + recall)

best_idx = np.argmax(f1_scores[:-1])  # Índice del mejor F1-score (evita NaN)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

fig, axes = plt.subplots(ncols=2, figsize=(20, 6))
axes[0].hist(y_pred_probs[test_labels == digito, digito], bins=20, alpha=0.5)
axes[0].set_title(f"Distribución de Probabilidades Predichas para el digito {digito}")
axes[0].set_xlabel("Probabilidad predicha")
axes[0].set_ylabel("Frecuencia")

axes[1].plot(recall, precision, label=f'F1 max = {best_f1:.2f}')
axes[1].scatter(recall[best_idx], precision[best_idx], marker='o', color='black')
axes[1].set_title(f"Curva Precisión-Recall para el digito {digito}")
axes[1].legend()
st.pyplot(plt)


# 4️⃣ **Análisis de Errores**: Ejemplos mal clasificados
errores_idx = np.where(y_pred_classes != test_labels)[0]  # Índices de errores
num_ejemplos = min(20, len(errores_idx))  # Mostrar hasta 5 errores

_, (imagenes_test, _) = mnist.load_data()

filas = 4
columnas = 5

errores_filtrados = [idx for idx in errores_idx if test_labels[idx] == digito]

num_ejemplos = min(20, len(errores_filtrados))
filas, columnas = 4, 5

plt.figure(figsize=(15, 6))
for i, idx in enumerate(errores_filtrados[:filas * columnas]):
    plt.subplot(filas, columnas, i + 1)
    plt.imshow(imagenes_test[idx].squeeze(), cmap='gray')
    plt.title(f"Real: {test_labels[idx]}\nPred: {y_pred_classes[idx]}")
    plt.axis("off")

plt.suptitle(f"Ejemplos mal clasificados para {digito}")
plt.tight_layout()
st.pyplot(plt)