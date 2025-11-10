# Stroke Prediction — Zero to Hero (Comprehensive Project Guide)

> This document explains every step of the Stroke Prediction project — from dataset to deployment — in detail. Place this file in your repository root so reviewers, collaborators, or hiring managers can fully understand what you did, why you did it, and how to reproduce or improve it.

---

# Table of Contents
1. Project Overview
2. Motivation & Use Case
3. Repository Structure
4. Dataset Details
5. Data Understanding & Exploratory Steps
6. Data Cleaning & Preprocessing (step-by-step)
7. Encoding Strategy (why & how)
8. Handling Class Imbalance (SMOTE) — reasoning and implementation
9. Feature Scaling
10. Train / Test Split — best practices
11. Models Trained (list + explanation + hyperparameters)
12. ANN architecture and training details
13. Model Evaluation & Metrics — why we choose them
14. Model Selection: Recall vs F1 — decision and implications
15. Saving artifacts (pickle / h5) — what to save and why
16. Streamlit Frontend — design and behavior
17. Reproducibility & Environment
18. How to run (step-by-step) — from training to frontend
19. Example inputs & expected outputs (test cases)
20. Diagnostics, Debugging, and Common Pitfalls
21. Limitations, Ethical Considerations & Medical Safety
22. Next steps & Suggestions for improvement
23. References & Credits

---

# 1. Project Overview
**What**: An end-to-end machine learning pipeline to predict the risk of stroke from patient-level features using multiple classical classifiers and an Artificial Neural Network (ANN). The pipeline includes preprocessing, handling class imbalance, model training, evaluation, saving the best model, and a Streamlit app for interactive predictions.

**Why**: Stroke is a high-impact health event. Early identification of high-risk individuals can prompt diagnostic tests, monitoring, and prevention. In this project we prioritize *sensitivity (recall)* to reduce false negatives — missing a stroke is more dangerous than producing extra precautionary tests.

---

# 2. Motivation & Use Case
- **Use case**: Screening tool to flag individuals who may require urgent medical follow-up. NOT a diagnostic tool.
- **Target audience**: Data scientists, ML engineers, healthcare data teams, and mentors reviewing your project.
- **Constraints**: This model is trained on a single dataset (Kaggle stroke dataset). Real clinical deployment would require rigorous validation, ethical approval, and integration with clinical workflows.

---

# 3. Repository Structure (recommended)
```
stroke_prediction/
├── backend_training.py       # Full pipeline: preprocess, train, evaluate, save
├── app.py                    # Streamlit frontend to load pickled files and predict
├── stroke-data.csv           # Dataset (or link to it) - not always committed for privacy
├── best_model.pkl or .h5     # Best model saved by pipeline
├── column_transformer.pkl    # ColumnTransformer (OneHotEncoder + passthrough)
├── scaler.pkl                # StandardScaler object
├── requirements.txt          # Python packages & versions
├── PROJECT_DOCUMENTATION.md  # This file (explain everything)
└── README.md                 # Short intro (optional)
```

---

# 4. Dataset Details
**Source:** Kaggle — "Stroke Prediction Dataset" (fedesoriano) or local CSV (`stroke-data.csv`).

**Columns** *(typical)*:
- `id` — unique patient id (not used as feature)
- `gender` — Male / Female / Other
- `age` — continuous
- `hypertension` — 0/1
- `heart_disease` — 0/1
- `ever_married` — Yes/No
- `work_type` — categorical (Private, Self-employed, Govt_job, children, Never_worked)
- `Residence_type` — Urban / Rural
- `avg_glucose_level` — continuous
- `bmi` — continuous (contains missing values)
- `smoking_status` — formerly smoked / never smoked / smokes / Unknown
- `stroke` — target (0 = No, 1 = Yes)

Important: real dataset may differ slightly in column names; always check `dataset.columns`.

---

# 5. Data Understanding & Exploratory Steps
1. **Check shape & head**: `dataset.shape`, `dataset.head()`
2. **Value counts** for categorical columns: `dataset['gender'].value_counts()` — identify rare categories (e.g., 'Other').
3. **Missing values**: `dataset.isnull().sum()` — BMI commonly missing.
4. **Class imbalance check**: `dataset['stroke'].value_counts()` — typically highly imbalanced (e.g., 95%:5%).
5. **Summary stats** for continuous features: `dataset.describe()`
6. **Correlation & pairplots** (optional) — to visualize relationships.

Record all insights in the notebook — these justify later choices (SMOTE, scaling, feature handling).

---

# 6. Data Cleaning & Preprocessing (step-by-step)
**1. Remove invalid categories**: e.g., `dataset = dataset[dataset['gender'] != 'Other']` — drop if only one or very few rows and not useful.

**2. Impute missing values**:
- `dataset['bmi'].fillna(dataset['bmi'].mean(), inplace=True)` — simple mean imputation; alternatives: median, KNN imputation.

**3. Feature selection**:
- Drop `id` column: non-informative for modeling.
- Keep all other features (domain knowledge may suggest excluding some, but we keep them for the model to decide importance).

**4. Save dataset snapshot** used for training (store hash or filename) for reproducibility.

---

# 7. Encoding Strategy
**Binary categorical features** (2 classes): use `LabelEncoder` (or manual mapping) — for columns like `gender`, `ever_married`, `Residence_type`.

**Multi-class categorical features**: use `OneHotEncoder` with `drop='first'` to avoid dummy variable trap. Implement with `ColumnTransformer`:
```python
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['work_type', 'smoking_status'])], remainder='passthrough')
X = ct.fit_transform(dataset.drop(['id','stroke'], axis=1))
```
**Why ColumnTransformer?** Keeps pipeline tidy and lets you apply different transformations to different columns with a single call.

**Note:** After `fit_transform`, `X` becomes a NumPy array — convert back to DataFrame for inspection if needed using `ct.get_feature_names_out()`.

---

# 8. Handling Class Imbalance — SMOTE
**Problem**: Target distribution heavily skewed towards `0` (no stroke). Models tend to predict the majority class.

**Solution applied**: SMOTE (Synthetic Minority Over-sampling Technique) applied *only* to the training set (some variants: oversampling before split, but prefer after splitting to avoid leakage).

Implementation pattern used in this project (applied to training set):
```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
```
**Why SMOTE?** Generates synthetic minority-class samples by interpolating between minority neighbors — helps models learn decision boundaries.

**Note & Caution:** SMOTE can create unrealistic samples if feature space is sparse; it may cause over-sensitivity. Always evaluate results and tune parameters.

Alternative approaches:
- Class weights (`class_weight='balanced'` in some estimators)
- Random oversampling or undersampling
- Advanced methods: ADASYN, SMOTEENN, ensemble methods

---

# 9. Feature Scaling
**StandardScaler** is used (z-score normalization) because some algorithms (SVM, KNN, ANN) are sensitive to feature scales.

Crucial detail: **fit the scaler on training data only**, then `transform` both train and test sets:
```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

We save the scaler using `pickle` so the same transformation is applied to new examples in the Streamlit app.

---

# 10. Train / Test Split
Use `train_test_split(X, y, test_size=0.2, random_state=0)` to hold out a test set for final evaluation. Important to split before scaling to prevent leakage (scaler fit on training only).

---

# 11. Models Trained
We train a diverse collection of models to compare performance and robustness:

- **Logistic Regression** (linear baseline, interpretable)
- **K-Nearest Neighbors (KNN)** (distance based)
- **Support Vector Machine (SVM)** (kernel based)
- **Decision Tree** (interpretable rule-based)
- **Random Forest** (bagging ensemble)
- **Gradient Boosting** (sklearn's GradientBoostingClassifier)
- **XGBoost** (optimized gradient boosting)
- **LightGBM** (fast gradient boosting)
- **Naive Bayes** (probabilistic baseline)

Each model is trained with reasonable defaults. For some we use `class_weight='balanced'` to further mitigate imbalance.

**How we evaluate:** After training, we compute: Accuracy, Precision, Recall, F1 score.

---

# 12. ANN (Neural Network) Details
ANN architecture used:
```python
ann = Sequential([
    Dense(8, activation='relu', input_dim=X_train.shape[1]),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size=32, epochs=50)
```
- Binary cross-entropy loss
- Sigmoid output for binary classification
- Trained alongside classical models for comparison

Tweakable hyperparameters: number of layers, units, activation functions, epochs, batch size, learning rate.

---

# 13. Model Evaluation & Metrics — Why these?
- **Accuracy** — overall fraction correct (not reliable with imbalanced data)
- **Precision** — when model predicts stroke, how often is it correct
- **Recall (Sensitivity)** — of actual stroke cases, how many the model caught
- **F1 Score** — harmonic mean of precision and recall, good balance metric

**In healthcare, recall is often prioritized** because false negatives (missed stroke) are more harmful than false positives (extra checks).

We produce classification reports and confusion matrices for the chosen best model.

---

# 14. Model Selection: Recall vs F1
- **Initial choice**: select best model by **Recall** to prioritize catching all positives. This produced a model that flagged more positives (higher sensitivity) but also increased false positives.

- **Tradeoff**: For practical applications you may prefer **F1** or a tuned threshold. We include guidance on how to change selection metric (modify the `results_df.sort_values(by='Recall')` line to `by='F1'` or implement threshold tuning).

Recommendation: Use Recall for clinical screening algorithms (with caveats), but show probability and let clinicians decide actions.

---

# 15. Saving Artifacts
We save objects required for deployment:
- `column_transformer.pkl` — OneHotEncoder + passthrough mapping
- `scaler.pkl` — StandardScaler fitted on training set
- `best_model.pkl` — pickled sklearn model OR `best_model.h5` for ANN

Why save preprocessors? So the frontend applies the exact same transformations as training — essential for consistent predictions.

---

# 16. Streamlit Frontend
**app.py** features:
- Loads `best_model.h5` (ANN) or `best_model.pkl` (sklearn) dynamically
- Loads `column_transformer.pkl` and `scaler.pkl`
- Provides form for inputs: gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status
- Applies the saved transformations and scaling
- Predicts and returns `High Risk` or `Low Risk` and optionally the probability

Important notes for frontend:
- **Do NOT** use SMOTE in frontend — SMOTE is for training only.
- Show probability and keep a threshold control for live tuning.

---

# 17. Reproducibility & Environment
**Recommended `requirements.txt`** (example):
```
python>=3.8
pandas
numpy
scikit-learn
tensorflow
xgboost
lightgbm
imblearn
streamlit
matplotlib
```
Use `pip install -r requirements.txt` or create a virtual environment.

**Reproducibility tips**:
- Fix random seeds where possible (`random_state=0`)
- Save the exact dataset snapshot used (or its checksum)
- Save trained models and preprocessors

---

# 18. How to run (step-by-step)
**Train & save models**
```bash
python backend_training.py
# or run the training notebook cell-by-cell
```

**Run Streamlit frontend**
```bash
streamlit run app.py
```

**Quick test**: use the example low-risk and high-risk inputs provided below.

---

# 19. Example Inputs & Expected Outputs (Test Cases)
**Low risk example**
```
Gender: Female
Age: 25
Hypertension: No
Heart Disease: No
Ever Married: No
Work Type: Private
Residence: Urban
Avg Glucose Level: 95
BMI: 22
Smoking Status: never smoked
```
Expected: `Low Risk` (most models)

**High risk example**
```
Gender: Male
Age: 68
Hypertension: Yes
Heart Disease: Yes
Ever Married: Yes
Work Type: Never_worked
Residence: Urban
Avg Glucose: 200
BMI: 32
Smoking Status: formerly smoked
```
Expected: `High Risk` (most models)

**Note**: If model returns unexpected results, check which trained model is loaded and whether SMOTE/class_weight were applied.

---

# 20. Diagnostics & Troubleshooting
1. **NameError: best_model not defined** — ensure `best_model` is assigned in both ANN and sklearn branches.
2. **Old pickle used in frontend** — delete old `.pkl`/`.h5` files and re-run backend to create fresh artifacts.
3. **Unexpected high-risk predictions for low-risk profile** — check: (a) glucose is high, (b) model selected by Recall, (c) threshold set to 0.5, (d) SMOTE effect. Use probability and threshold tuning.
4. **ColumnTransformer errors on transform** — ensure feature order and names match training; use `ct.get_feature_names_out()` to inspect.

---

# 21. Limitations, Ethics & Medical Safety
- **NOT a diagnostic tool**: This is a prototype. Any real deployment requires clinical validation and regulatory approval.
- **Bias risk**: Even with SMOTE, dataset bias (collection bias, demographic imbalance) can persist. Report subgroup metrics (recall by age, gender) before deployment.
- **Explainability**: Use SHAP for feature attribution if using model for clinical decisions.
- **Privacy**: Do not include personally identifiable data in public repos.

---

# 22. Next steps & Improvements
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for each model
- Cross-validation and nested CV for unbiased model selection
- Use SHAP or LIME for explanations
- Expand dataset or combine multiple datasets
- Calibrate probabilities (Platt scaling / isotonic) if needed
- Explore ensemble stacking or meta-model selection

---

# 23. References & Credits
- Kaggle: Stroke Prediction Dataset by fedesoriano
- scikit-learn documentation
- imbalanced-learn (SMOTE) documentation
- TensorFlow / Keras documentation

---

# Final Note
This project is intentionally designed as a **teaching and demonstration** tool: it shows a complete ML lifecycle in a sensitive domain (healthcare). The code is modular — you can retrain, swap models, change evaluation metric, and deploy safely. Keep detailed experimental logs (parameters, metrics) to support reproducibility and auditability.

Good luck — you’ve built something meaningful and challenging! 🎯

---

*End of document.*

