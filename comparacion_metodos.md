# Comparación de Métodos de Codificación Categórica

En el procesamiento del *Adult Census Income Dataset*, las variables categóricas (como `workclass`, `education` o `race`) son de tipo nominal u ordinal, lo que exige aplicar diferentes estrategias de codificación para que los algoritmos matemáticos de Machine Learning puedan procesarlas. A continuación se detallan las ventajas y desventajas de los tres métodos aplicados.

## 1. get_dummies (Pandas)
Este método crea una nueva columna binaria (0 o 1) para cada categoría única presente en la variable original.

* **Ventajas:** * Es la opción más rápida y sencilla para análisis exploratorio de datos (EDA).
  * Devuelve automáticamente un `DataFrame` de Pandas con los nombres de las columnas bien formateados, listo para ser visualizado sin pasos intermedios.
* **Desventajas:**
  * No está diseñado para entornos de producción de Machine Learning. Si en el futuro llegan datos nuevos (conjunto de test) que no contienen una categoría específica que sí estaba en el entrenamiento, `get_dummies` generará un número diferente de columnas, rompiendo el modelo.

## 2. OneHotEncoder (Scikit-Learn)
Al igual que `get_dummies`, genera codificación binaria en múltiples columnas, pero está integrado en el ecosistema de Scikit-Learn.

* **Ventajas:**
  * **Robustez en producción:** Guarda un estado interno de las categorías ("fit"). Al hacer `.transform()` sobre datos nuevos, garantiza que la salida siempre tendrá el mismo número de columnas, ignorando categorías desconocidas o rellenando con ceros si faltan.
  * Permite retornar matrices dispersas (sparse matrices), ahorrando una cantidad masiva de memoria RAM en datasets con alta cardinalidad.
* **Desventajas:**
  * Retorna un array de Numpy por defecto, no un DataFrame. Requiere código adicional (como `get_feature_names_out`) para reconstruir la tabla legible, lo que lo hace menos directo para análisis visual rápido.

## 3. LabelEncoder (Scikit-Learn)
Sustituye cada categoría por un valor entero secuencial (ej. "Blanco"=0, "Negro"=1, "Asiático"=2). El número de columnas del dataset no aumenta.

* **Ventajas:**
  * Es increíblemente eficiente en memoria, ya que no añade nuevas columnas al dataset.
  * Es la codificación obligatoria para la variable objetivo (la etiqueta o "Target", como la columna `income`).
  * Funciona perfectamente con modelos basados en árboles de decisión (Random Forest, XGBoost).
* **Desventajas:**
  * **Peligro matemático:** Introduce una falsa ordinalidad. Si aplicamos esto a la variable `workclass`, el modelo de Regresión Lineal o KNN interpretará matemáticamente que la categoría 2 es "el doble" que la categoría 1, lo cual es falso y arruinará las predicciones. Solo debe usarse para variables puramente ordinales o binarias (como `sex`).

  