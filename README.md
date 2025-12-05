# 🏠 Python – Deep Learning: Housing Price Prediction with Neural Networks

This repository contains a **Deep Learning project** for predicting housing prices using **fully connected neural networks** built with Keras/TensorFlow. The project includes feature engineering, regularization techniques (L1, L2, Dropout) to combat overfitting, and achieves strong predictive performance (RMSE ~38,421).

Este repositorio contiene un **proyecto de Deep Learning** para predecir precios de viviendas usando **redes neuronales densas** construidas con Keras/TensorFlow. El proyecto incluye ingeniería de características, técnicas de regularización (L1, L2, Dropout) para combatir el sobreajuste, y logra un rendimiento predictivo sólido (RMSE ~38,421).

Nota: este ejercicio fue realizado como caso final en la asignatura de máster y obtuvo la calificación máxima.
---

## 📄 Files / Archivos

- `notebooks/Artiles_Suarez_JoseMiguel_CasopracticoFinal_DL.ipynb` – Complete Jupyter notebook with full pipeline (EDA, feature engineering, model training, regularization experiments).  
- `data/train_v2.csv` – Training dataset with housing features and sale prices.  
- `data/test_v2.csv` – Test dataset for final predictions.  
- `images/` – Visualizations (loss curves, overfitting analysis, etc.).

---

## 🎯 Objectives / Objetivos

- Build a **regression model** to predict `SalePrice` based on housing features.  
- Apply **feature engineering** to create meaningful derived variables.  
- Handle **categorical variables** by mapping them to numeric scores.  
- Implement **neural networks** with multiple hidden layers using Keras.  
- Combat **overfitting** using regularization techniques: L1, L2, and Dropout.  
- Evaluate models using **RMSE** (Root Mean Squared Error) as the loss function.

- Construir un **modelo de regresión** para predecir `SalePrice` en función de características de viviendas.  
- Aplicar **ingeniería de características** para crear variables derivadas significativas.  
- Manejar **variables categóricas** mapeándolas a puntuaciones numéricas.  
- Implementar **redes neuronales** con múltiples capas ocultas usando Keras.  
- Combatir el **sobreajuste** usando técnicas de regularización: L1, L2 y Dropout.  
- Evaluar modelos usando **RMSE** (Error Cuadrático Medio) como función de pérdida.

---

## 🛠️ Tech stack / Tecnologías utilizadas

- **Python 3.x**  
- **TensorFlow 2.3.0** & **Keras 2.3.1** – Deep learning framework.  
- **NumPy 1.18.5** – Numerical operations.  
- **Pandas** – Data manipulation and cleaning.  
- **Matplotlib** – Loss curve visualization.  
- **scikit-learn** – Train-test split, MinMaxScaler.  
- **Google Colab** – Cloud-based Jupyter environment.

---

## 🧹 Workflow / Flujo de trabajo

### 1. Data loading & EDA / Carga de datos y análisis exploratorio

- Load `train_v2.csv` (1,460 houses, 81 features) and `test_v2.csv`.  
- Check for missing values, duplicates and basic statistics.  
- Identify categorical vs numerical features.

### 2. Feature engineering / Ingeniería de características

**Created variables / Variables creadas:**

- **`House_Age`**: `YrSold - YearBuilt` (age of the house at sale).  
- **`House_Newness_Score`**: `exp(-k * House_Age)` – Exponential decay score favoring newer houses (k=0.05).  
- **`SaleConditionImpacto`**: Numeric mapping of sale conditions (Normal=1.0, Abnormal=-0.7, AdjLand=1.5, etc.).

### 3. Categorical to numeric transformation / Transformación de categóricas a numéricas

Manual mapping of categorical variables to numeric scores based on domain knowledge:

- **Material quality** (e.g., `Exterior1st`, `MasVnrType`): BrkFace=5, Stone=4, Plywood=1, etc.  
- **Condition scores** (e.g., `ExterQual`, `KitchenQual`): Ex=5, Gd=4, TA=3, Fa=2, Po=1.  
- **Garage/Basement quality**: Similar mapping for `GarageType`, `GarageFinish`, `BsmtQual`, etc.  
- **Sale characteristics**: `SaleType`, `SaleCondition` mapped to numeric relevance scores.

### 4. Data cleaning / Limpieza de datos

- **Dropped columns with excessive NaNs**: `Alley`, `PoolQC`, `Fence`, `FireplaceQu`, `MiscFeature`, `Utilities`, `LotFrontage`, etc.  
- **Removed redundant variables**: `YearBuilt`, `YrSold` (replaced by `House_Age` and `House_Newness_Score`).  
- **Eliminated remaining categorical columns** after numeric transformation.  
- **Dropped columns with any NaN** to ensure clean input for neural networks.

### 5. Train-validation split / División entrenamiento-validación

- **80% training, 20% validation** using `train_test_split` (random_state=42).  
- Final split: `X_train` (1,168 samples), `X_val` (292 samples).

### 6. Min-Max normalization / Normalización Min-Max

- Applied `MinMaxScaler` to scale all numeric features to [0, 1] range.  
- Prevents features with large ranges from dominating the neural network learning.

### 7. Neural network architecture / Arquitectura de red neuronal

**Baseline model:**
- **Input layer**: 50 features (after cleaning).  
- **Hidden layer 1**: 200 neurons, ReLU activation.  
- **Hidden layer 2**: 200 neurons, ReLU activation.  
- **Output layer**: 1 neuron (regression, no activation).  
- **Optimizer**: Adam (learning_rate=0.001).  
- **Loss function**: MSE (Mean Squared Error).  
- **Metrics**: RMSE (custom function).

**Training:**
- 100 epochs, batch_size=32.  
- **Final validation RMSE**: ~38,421.72 (strong improvement from initial ~100k).

### 8. Overfitting experiment / Experimento de sobreajuste

Created a deliberately **overfit model** to demonstrate the problem:

- **4 hidden layers**: 500, 500, 300, 200 neurons (excessive capacity).  
- **Result**: Training loss drops sharply, validation loss increases → clear overfitting signal.

### 9. Regularization experiments / Experimentos de regularización

Tested **3 regularization techniques** to combat overfitting:

1. **L2 (Ridge)**: Penalty proportional to squared weights (`kernel_regularizer=l2(0.01)`).  
2. **L1 (Lasso)**: Penalty proportional to absolute weights (`kernel_regularizer=l1(0.01)`), encourages sparsity.  
3. **Dropout**: Randomly deactivates 50% of neurons during training (`Dropout(0.5)`).

**Results:**
- All three techniques reduced overfitting.  
- **Best regularization**: [Mejor regularización: L1 con pérdida (MSE) = 1615437440.00].

---

## 📊 Key results / Resultados clave

- **Baseline model RMSE**: ~38,421 on validation set (strong predictive performance).  
- **Feature engineering impact**: `House_Newness_Score` and `SaleConditionImpacto` improved model interpretability and potentially reduced RMSE.  
- **Overfitting successfully demonstrated**: Large model (4 layers, 1500+ neurons) showed training loss << validation loss.  
- **Regularization effectiveness**: L1, L2, and Dropout all mitigated overfitting, with Dropout typically performing best.

---

## 🚀 How to run / Cómo ejecutar

1. Clone this repository:  
git clone https://github.com/JM-specialist-network/python-deep-learning-housing-prices.git
cd python-deep-learning-housing-prices

2. Install dependencies:
pip install tensorflow==2.3.0 keras==2.3.1 numpy==1.18.5 pandas matplotlib scikit-learn


3. Open the notebook in Jupyter or Google Colab:  
jupyter notebook notebooks/DeepLearning_practica.ipynb

4. Run all cells to reproduce the full pipeline (EDA → feature engineering → training → regularization experiments).

---

## 🔍 Key insights / Hallazgos principales

- **Domain-driven feature engineering matters**: Creating `House_Newness_Score` (exponential decay based on house age) captures intuitive real estate dynamics.  
- **Manual categorical encoding works well**: Mapping categories to numeric scores based on quality/relevance (e.g., Ex=5, Po=1) preserves ordinal relationships better than one-hot encoding for high-cardinality features.  
- **Neural networks benefit from normalization**: Min-Max scaling to [0, 1] accelerated convergence and improved final RMSE.  
- **Overfitting is controllable**: Even a model with 1500+ parameters can be regularized effectively with L1/L2/Dropout.  
- **RMSE ~38k on validation**: Given that median house price is ~$163,000, this represents ~23% error—reasonable for a complex real estate dataset.

---

## 📚 Technical highlights / Aspectos técnicos destacados

- **Custom RMSE metric**: Implemented custom Keras metric using `K.sqrt(K.mean(K.square(...)))` to track RMSE during training.  
- **Systematic data cleaning**: Removed 20+ redundant or high-NaN columns, reducing feature space from 81 to 50 features.  
- **Visualization of overfitting**: Plotted training vs validation loss curves to demonstrate the gap in overfit model.  
- **Comparative regularization study**: Quantitatively compared L1, L2, and Dropout on identical architecture.
- **Integration of knowledge of business**: for create dummy variables, key is business!
---

## 🎓 Academic context / Contexto académico

Demonstrates mastery of:
- Neural network architecture design.  
- Feature engineering for tabular data.  
- Overfitting diagnosis and mitigation.  
- Regression with deep learning (vs traditional ML).

---

## 👤 Author / Autor

Created by **Jose Miguel Artiles** – Data Scientist & Economist-in-training.  

- GitHub: [JM-specialist-network](https://github.com/JM-specialist-network)  
- Email: joseartiles@g***l.com

