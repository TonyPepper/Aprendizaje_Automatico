import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

df = pd.read_csv("weatherAUS.csv", delimiter=",")

def FunctionTransformer(df):#funcion para actualizar dataframe
  dff=df.copy()
  localidades = ["SydneyAirport", "Sydney", "Canberra", "Melbourne", "MelbourneAirport" ]

  dff= dff[dff['Location'].isin(localidades)]
  dff = dff.dropna(subset=['RainTomorrow', 'RainfallTomorrow'])

  dff = df.drop('Location', axis=1)
  def rellenar(t):
      for columna in t.columns:

          # Verifica si la columna tiene datos faltantes
          if t[columna].isnull().any():

              # Si es numérica y continua, rellena con el promedio
              if pd.api.types.is_numeric_dtype(t[columna]) and not pd.api.types.is_integer_dtype(t[columna]):
                  t[columna].fillna(t[columna].mean(), inplace=True)

              # Si es numérica y discreta, rellena con la moda
              elif pd.api.types.is_numeric_dtype(t[columna]):
                  t[columna].fillna(t[columna].mode()[0], inplace=True)

            # Si es categórica, rellena con la moda
              else:
                  t[columna].fillna(t[columna].mode()[0], inplace=True)

      return t
  dff = rellenar(dff)
  dff =  dff.reset_index(drop = True)
  columnas_categoricas = dff.select_dtypes(exclude='number').columns
  le = LabelEncoder()
  for categorica in columnas_categoricas:
    dff[categorica] = le.fit_transform(dff[categorica])
  dff = dff.replace([np.inf, -np.inf], 0)
  dff.drop('Unnamed: 0', axis=1, inplace=True)
  dff.drop('Date', axis=1, inplace=True)


  return dff

dff=FunctionTransformer(df)

#------------# Datos #------------#

regresion = dff.copy()
df_regresion = regresion.drop(['RainTomorrow', 'RainfallTomorrow'], axis=1)
input_shape= (len(regresion.columns)-2,)
feature_names = regresion.columns
#------------# Datos regresion #------------#

# Dividir los datos en conjunto de entrenamiento y de prueba
X_train_R, X_test_R, Y_train_R, Y_test_R = train_test_split(
    df_regresion,
   dff['RainfallTomorrow'].values.reshape(-1, 1),
    test_size=0.2,
    random_state=12)

#------------# Datos clasificacion #------------#

# Dividir los datos en conjunto de entrenamiento y de prueba
X_train_C, X_test_C, Y_train_C, Y_test_C = train_test_split(
    df_regresion,
    dff['RainTomorrow'].values.reshape(-1, 1),
    test_size=0.2,
    random_state=12)
X = dff.drop(['RainTomorrow', 'RainfallTomorrow'], axis=1)
y = dff['RainTomorrow']


#------------# Redes Neuronales #------------#

# Definir y compilar el modelo de red neuronal
model_clasificacion = Sequential([
    tf.keras.layers.Dense(8, activation='relu'),
     tf.keras.layers.Dense(33, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_clasificacion.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.Precision(thresholds=0.7)])
# Define el modelo para regresión
model_regresion2 = tf.keras.Sequential([
    tf.keras.layers.Dense(25, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(65, activation='relu'),
    tf.keras.layers.Dense(34, activation='relu'),
    tf.keras.layers.Dense(1)  # No se utiliza la función de activación en la capa de salida para regresión
])

# Compila el modelo para regresión
model_regresion2.compile(optimizer='adam',
                        loss='mean_squared_error',  # Utiliza la pérdida de error cuadrático medio para regresión
                        metrics=['mean_absolute_error'])  # Puedes ajustar las métricas según tus necesidades

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # esto simplemente imputa por la media.
    ('scaler', StandardScaler()),  # se realiza una estandarización.
    ('model', model_clasificacion),  # se entrena prediccion 'RainTomorrow'
])


pipeline2 = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # esto simplemente imputa por la media.
    ('scaler', StandardScaler()),  # se realiza una estandarización.
    ('model 2', model_regresion2) # se entrena prediccion 'RainfallTomorrow'
])

#------------# Tuberias #------------#
pipeline.fit(X_train_C, Y_train_C)#clasificacion
pipeline2.fit(X_train_R, Y_train_R)#regresion

#------------# Guardar modelos #------------#
joblib.dump(pipeline, 'clima_pipeline.joblib')
joblib.dump(pipeline2, 'clima_pipeline_regre.joblib')