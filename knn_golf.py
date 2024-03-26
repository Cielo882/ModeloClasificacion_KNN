import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Variables globales para almacenar los datos de entrenamiento
X_entrenamiento_global = None
y_entrenamiento_global = None

# Funcion para cargar datos desde un archivo seleccionado en la GUI
def cargar_datos():
    archivo_ruta = filedialog.askopenfilename(filetypes=[("Archivos de texto", "*.txt")])
    if archivo_ruta:
        datos = pd.read_csv(archivo_ruta, header=None)
        # Convertir a minúsculas y luego reemplazar etiquetas "yes" con 1 y etiquetas "no" con 0
        datos.iloc[:, -1] = datos.iloc[:, -1].str.lower().replace({"yes": 1, "no": 0})
        # Eliminar filas con valores faltantes (NaN) en la última columna
        datos = datos.dropna(subset=[datos.columns[-1]])
        return datos.values
    return None

# Funcion para dividir los datos en conjuntos de entrenamiento y prueba
def dividir_datos(X, y, tamano_prueba,semilla_aleatoria):
    np.random.seed(semilla_aleatoria)
    n = X.shape[0]
    indices = np.random.permutation(n)                                                                                                  
    indice_division = int((1 - tamano_prueba) * n)
    indices_entrenamiento = indices[:indice_division]
    indices_prueba = indices[indice_division:]
    X_entrenamiento, X_prueba = X[indices_entrenamiento], X[indices_prueba]
    y_entrenamiento, y_prueba = y[indices_entrenamiento], y[indices_prueba]
    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba

# Funcion para calcular la distancia euclidiana entre dos puntos
def distancia_euclidiana(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Funcion para predecir la clase de una instancia utilizando el clasificador KNN
def predecir_knn(X_entrenamiento, y_entrenamiento, X_prueba, k):
    y_predicho = []
    for i in range(X_prueba.shape[0]):
        distancias = [(distancia_euclidiana(X_prueba[i], x), y) for x, y in zip(X_entrenamiento, y_entrenamiento)]
        distancias.sort(key=lambda x: x[0])
        vecinos_mas_cercanos = distancias[:k]
        etiquetas_vecinos = [vecino[1] for vecino in vecinos_mas_cercanos]
        mas_comun = np.bincount(etiquetas_vecinos).argmax()
        y_predicho.append(mas_comun)
    return np.array(y_predicho)

# Crear la ventana de la GUI
ventana = tk.Tk()
ventana.title("Carga de Datos")

# Obtener las dimensiones de la ventana
ancho_ventana = ventana.winfo_reqwidth()
alto_ventana = ventana.winfo_reqheight()

# Obtener las dimensiones de la pantalla y calcular el centro
ancho_pantalla = ventana.winfo_screenwidth()
alto_pantalla = ventana.winfo_screenheight()
x = (ancho_pantalla/2) - (ancho_ventana/2)
y = (alto_pantalla/2) - (alto_ventana/2)

# Centrar la ventana en la pantalla
ventana.geometry(f'+{int(x)}+{int(y)}')

# Almacenar las precisiones promedio para cada valor de k
precisiones = []
valores_k = [3, 5, 7]

# Lista para almacenar las precisiones promedio
precisiones_promedio = []

# Funcion para entrenar el modelo
def entrenar():
    global X_entrenamiento_global, y_entrenamiento_global
    datos = cargar_datos()
    if datos is not None:
        X = datos[:, :-1]
        y = datos[:, -1]
        tamano_prueba=0.2
        semilla_aleatoria=None

        for k in valores_k:
            resultados = []
            for _ in range(20):
                X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = dividir_datos(X, y, tamano_prueba, semilla_aleatoria)
                y_predicho = predecir_knn(X_entrenamiento, y_entrenamiento, X_prueba, k)
                exactitud = np.mean(y_predicho == y_prueba) * 100
                resultados.append(exactitud)
            
            promedio_rendimiento = np.mean(resultados)
            precisiones.append(promedio_rendimiento)

            X_entrenamiento_global = X_entrenamiento
            y_entrenamiento_global = y_entrenamiento

        mejor_k = valores_k[np.argmax(precisiones)]
        print(f"El mejor valor de k es {mejor_k} con una precisión promedio de {max(precisiones):.2f}%")

        precisiones_promedio.extend(precisiones)

        plt.figure(figsize=(8, 6))
        plt.bar(valores_k, precisiones_promedio, tick_label=valores_k)
        plt.xlabel("Valor de k")
        plt.ylabel("Precisión Promedio")
        plt.title("Precisión Promedio vs. Valor de k")
        plt.show()

# Funcion para cargar un segundo archivo y realizar predicciones con el mejor modelo
def cargar_segundo_archivo():
    archivo_ruta = filedialog.askopenfilename(filetypes=[("Archivos de texto", "*.txt")])
    if archivo_ruta:
        datos = pd.read_csv(archivo_ruta, header=None)
        X = datos.values
        
        mejor_k = valores_k[np.argmax(precisiones)]
        y_predicho = predecir_knn(X_entrenamiento_global, y_entrenamiento_global, X, mejor_k)

        etiqueta = np.array(["yes" if i == 1 else "no" for i in y_predicho])
        
        print(f"Predicciones para el segundo archivo con k={mejor_k}:")
        print(etiqueta)

boton_entrenar = tk.Button(ventana, text="Entrenar Modelo y Graficar", command=entrenar)
boton_entrenar.pack()

boton_cargar_segundo_archivo = tk.Button(ventana, text="Cargar Segundo Archivo y Predecir", command=cargar_segundo_archivo)
boton_cargar_segundo_archivo.pack()

ventana.mainloop()
