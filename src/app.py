# # LINDA VASQUEZ Random forest

# ## 1. Carga de datos

import pandas as pd

link_entrenamiento = "https://raw.githubusercontent.com/lindavasquez28/LIVASQUE_proyecto-derbol_de_decision/refs/heads/main/data/processed/clean_diabetes_train.csv"
link_prueba = "https://raw.githubusercontent.com/lindavasquez28/LIVASQUE_proyecto-derbol_de_decision/refs/heads/main/data/processed/clean_diabetes_test.csv"

datos_entrenamiento = pd.read_csv(link_entrenamiento)
datos_prueba = pd.read_csv(link_prueba)

x_entrena = datos_entrenamiento.drop(["Outcome"], axis = 1)
y_entrena = datos_entrenamiento["Outcome"]

x_prueba = datos_prueba.drop(["Outcome"], axis = 1)
y_prueba = datos_prueba["Outcome"]

datos_entrenamiento.head()

# ## 2. Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Creacion del modelo
modelo_bosque = RandomForestClassifier(n_estimators = 60, random_state = 42)

# Entrenamiento
modelo_bosque.fit(x_entrena, y_entrena)

#Predicciones
predicciones = modelo_bosque.predict(x_prueba)

# Calculo de precisión
precision = accuracy_score(y_prueba, predicciones)

print (predicciones)
print(f"\nLa precisión del Random Forest es: {precision}")

# ## 3. Guardado de modelo

from pickle import dump

dump(modelo_bosque, open("../models/bosque_aleatorio_n60_42.sav", "wb"))
dump(modelo_bosque, open("../data/bosque_aleatorio_n60_42.sav", "wb"))

print("Modelo guardado en /models/bosque_aleatorio_n60_42.sav")
print("Modelo guardado en /data/bosque_aleatorio_n60_42.sav")

