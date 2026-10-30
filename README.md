# Modelado Predictivo de Churn Rate: Ingeniería de ML para la Retención de Clientes en Beta Bank

![Python](https://shields.io)
![Scikit-Learn](https://shields.io)
![Pandas](https://shields.io)
![Status](https://shields.io)

## 1. Introducción y Planteamiento del Problema
En el sector bancario, la pérdida de clientes (*churn rate*) representa uno de los desafíos financieros más críticos para la rentabilidad institucional. Debido a que los costos asociados con la adquisición de un nuevo usuario superan considerablemente a los de retención, **Beta Bank** requiere una estrategia proactiva para mitigar este fenómeno.

El objetivo principal de este proyecto es construir un pipeline de Machine Learning y un modelo de Clasificación Binaria capaz de predecir, con alta precisión, qué cuentahabientes tienen una probabilidad elevada de abandonar el banco en el corto plazo, sirviendo como una herramienta de optimización financiera directamente aplicable al control operativo del banco.

---

## 2. Flujo de Trabajo Estandarizado (Data Science Rigor)

El proyecto se estructuró bajo un flujo operativo secuencial de cinco fases interdependientes para neutralizar sesgos metodológicos:

*   **Fase 1: Auditoría y Curación de Datos (Data Wrangling):** Remoción de variables administrativas redundantes (`RowNumber`, `CustomerId`, `Surname`). Transformación categórica mediante mapeo binario (`Gender`) y One-Hot Encoding (`Geography`) previniendo la trampa de la variable ficticia (*dummy variable trap*).
*   **Fase 2: Segmentación y Control de Filtraciones (Data Leakage):** División tripartita estructurada del dataset en **Entrenamiento (60%)**, **Validación (20%)** y **Prueba Final (20%)**. Implementación consecutiva de partición estratificada (`stratify=y`) para mitigar la asimetría original. La imputación de la variable `Tenure` (mediana = 5.0) y el escalado de características continuas (`StandardScaler`) se entrenaron **únicamente** con el set de entrenamiento para blindar el pipeline.
*   **Fase 3: Establecimiento de Modelos de Línea Base:** Entrenamiento nativo de arquitecturas lineales (Regresión Logística) y no lineales (Árbol de Decisión) ignorando el desequilibrio de clases para documentar el sesgo algorítmico y la insuficiencia del umbral por defecto (0.50).
*   **Fase 4: Optimización Avanzada por Desbalanceo:** Aplicación y contraste de dos enfoques de corrección: Ajuste de pesos en la función de coste (`class_weight='balanced'`) y modificación física de densidades por remuestreo estadístico (`Upsampling`). Sintonización iterativa de la complejidad estructural controlando el hiperparámetro de profundidad máxima (`max_depth`).
*   **Fase 5: Sintonización de Umbral y Evaluación Definitiva:** Búsqueda del umbral operativo óptimo de forma 100% ciega sobre las probabilidades en el conjunto de Validación. Evaluación, por primera y única vez, en el conjunto de Prueba Final aislado para validar la capacidad real de generalización.

---

## 3. Resultados y Desempeño Comercial

### Comparativa de Modelos en el Entorno de Validación
Debido a la asimetría estructural de las clases (~80% permanencia vs. ~20% abandono), la métrica de *Accuracy* global fue descartada como indicador de éxito, guiando la optimización mediante la media armónica del **F1-Score** (meta mínima de aprobación ≥ 0.59) y el **AUC-ROC**.

| Arquitectura / Enfoque | Técnica de Balanceo | Profundidad (`max_depth`) | F1-Score | AUC-ROC | Estado |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Regresión Logística | Línea Base (Ninguna) | Naive | 0.3160 | 0.7562 | Insuficiente |
| Árbol de Decisión | Línea Base (Ninguna) | Sin límite (Overfitting) | 0.5024 | 0.6884 | Insuficiente |
| Regresión Logística | Pesos de Clase Balanced | Naive | 0.4800 | 0.7607 | Insuficiente |
| Árbol de Decisión | Pesos de Clase Balanced | 6 | 0.5880 | - | Pre-aprobado |
| Regresión Logística | Upsampling | Naive | 0.4823 | 0.7622 | Insuficiente |
| **Árbol de Decisión (Campeón)** | **Upsampling** | **7** | **0.5941** | **0.7003** | **APROBADO** |

### Evaluación Definitiva Ex-Post (Conjunto de Prueba Final)
Evaluado sobre el set de control ciego aplicando el umbral sintonizado de **0.60**, el modelo final arrojó magnitudes sólidas y metodológicamente consistentes:
*   **F1-Score Final en Prueba:** `0.6011` (Superando de forma legítima la meta regulatoria de 0.59)
*   **AUC-ROC Final en Prueba:** `0.8234` (Excelente consistencia y ordenamiento probabilístico)
*   **Exactitud Global (Accuracy):** `82.45%`

---

## 4. Matriz de Confusión y Retorno de Inversión (ROI)

La distribución analítica absoluta de las predicciones sobre los 2,000 registros independientes del conjunto de prueba final demuestra la viabilidad comercial del sistema:

*   **Verdaderos Negativos (1,386 | Especificidad: 87.0%):** Identificación precisa de clientes estables, minimizando falsas alarmas operativas y protegiendo el presupuesto de marketing al evitar dirigir incentivos innecesarios a usuarios fidelizados.
*   **Verdaderos Positivos (263 | Recall: 64.62%):** El pipeline intercepta de manera oportuna a casi 65 de cada 100 cuentahabientes en riesgo legítimo de fuga, permitiendo al banco ejecutar campañas de retención altamente dirigidas con un ROI optimizado.

---

## 5. Estructura del Repositorio
```text
├── datasets/
│   └── Churn.csv             # Conjunto de datos original de Beta Bank
├── modelado_churn_bank.ipynb # Notebook principal con el pipeline y código limpio
├── README.md                 # Reporte ejecutivo del proyecto
└── .gitignore                # Exclusión de entornos virtuales y archivos residuales
```

---

## 6. Requisitos e Instalación
Para replicar de forma exacta los experimentos y las transformaciones vectoriales de este proyecto, asegúrate de contar con el entorno tecnológico configurado:

```bash
# Clonar el repositorio
git clone https://github.com

# Instalar dependencias estándar
pip install pandas numpy scikit-learn seaborn matplotlib
```

## 7. Ver el análisis completo
👉 [Haz clic aquí para ver el código y los gráficos interactivos](modelado_predictivo_churn_rate_beta_bank.ipynb)



