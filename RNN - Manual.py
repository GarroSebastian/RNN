import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Sequential
from keras import layers
from scipy.stats import mode

# Establecer la variable de entorno para desactivar oneDNN
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Cargar el dataset vectorizado
input_file_path = r'C:\Universidad\2024-1\Seminario 1\RNN\textos_vectorizados_tfidf_manual1.csv'
data = pd.read_csv(input_file_path)

# Asumir que la columna 'sentimiento' es el target
# Convertir 'sentimiento' a valores numéricos
# Negativo es 2 para que sea compatible con SparseCategoricalCrossEntropy
data['sentimiento'] = data['sentimiento'].map({'positivo': 1, 'neutral': 0, 'negativo': 2})

# Definir características (X) y etiqueta (y)
X = data.drop(columns=['Producto', 'Marca', 'Modelo', 'calificacion', 'Fecha', 'Texto', 'texto_limpio', 'sentimiento', 'tokens','dominant_topic', 'category'])
y = data['sentimiento']


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir a matrices NumPy
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Redimensionar para RNN (agregar una dimensión)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Verifica si hay NaN o valores nulos en las etiquetas
print(np.isnan(y_train).sum())
print(np.isnan(y_test).sum())

print(np.unique(y_train))
print(np.unique(y_test))

# Elimina las muestras con etiquetas NaN en y_train y y_test
X_train = X_train[~np.isnan(y_train)]
y_train = y_train[~np.isnan(y_train)]

X_test = X_test[~np.isnan(y_test)]
y_test = y_test[~np.isnan(y_test)]

print(np.unique(y_train))
print(np.unique(y_test))

# Crear el modelo RNN con capas adicionales y dropout
model = Sequential()
model.add(layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(layers.Dropout(0.3))
model.add(layers.LSTM(64))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# Compilar el modelo
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer= tf.keras.optimizers.Adam(), metrics=['accuracy'])
#model.compile(loss='SparsecategoricalEntropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# Añadir EarlyStopping para detener el entrenamiento cuando la validación no mejore
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test))

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Predicciones y métricas
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Verdadero')
plt.show()

# Reporte de clasificación
print(classification_report(y_test, y_pred, target_names=['Neutral', 'Positivo', 'Negativo'], zero_division=0))

# Curvas ROC para cada clase
fpr = {}
tpr = {}
roc_auc = {}
n_classes = 3
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Gráfico de todas las curvas ROC
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Clase {i} (área = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC para cada clase')
plt.legend(loc="lower right")
plt.show()

# Gráfico de la precisión a través de las épocas
plt.figure()
plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
plt.title('Precisión del modelo a través de las épocas')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()

# Resumen del sentimiento por tópicos
topic_sentiment_summary = data.groupby(['category', 'sentimiento']).size().unstack().fillna(0)
print(topic_sentiment_summary)

# Gráfico de barras del sentimiento por tópicos
plt.figure(figsize=(10, 6))
topic_sentiment_summary.plot(kind='bar', color=['red', 'blue', 'green'], ax=plt.gca())
plt.title('Resumen del Sentimiento por Tópicos')
plt.xlabel('Tópico')
plt.ylabel('Número de Reseñas')
plt.legend(title='Sentimiento', labels=['Negativo', 'Neutral', 'Positivo'], loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Guardar el modelo entrenado (opcional)
model.save(r'C:\Universidad\2024-1\Seminario 1\RNN\modelo_rnn_tfidf.h5')