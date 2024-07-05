import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import layers

# Configurar TensorFlow para utilizar la GPU
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Configured {len(physical_devices)} GPU(s) for use.")
    except RuntimeError as e:
        print(e)

# Cargar el dataset vectorizado
input_file_path = r'C:\Universidad\2024-1\Seminario 1\RNN\textos_vectorizados_tfidf.csv'
data = pd.read_csv(input_file_path)

# Asumir que la columna 'sentiment_VADER' es el target
# Convertir 'sentiment_VADER' a valores numéricos
data['sentiment_VADER'] = data['sentiment_VADER'].map({'positivo': 1, 'neutral': 0, 'negativo': -1})

# Definir características (X) y etiqueta (y)
X = data.drop(columns=['Producto', 'Marca', 'Modelo', 'CalificaciÃ³n', 'Fecha', 'Texto', 'texto_limpio', 'dominant_topic', 'category', 'scores', 'compound', 'sentiment_VADER'])
y = data['sentiment_VADER']

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

# Crear el modelo RNN
model = Sequential()
model.add(layers.LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(layers.Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Guardar el modelo entrenado (opcional)
model.save(r'C:\Universidad\2024-1\Seminario 1\RNN\modelo_rnn_tfidf.h5')