#Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
import pickle 
import tensorflow as tf
from sklearn.metrics import (confusion_matrix,accuracy_score,classification_report,f1_score,precision_score,recall_score)
from imblearn.over_sampling import SMOTE


#Load dataset and cleaning
dataset = pd.read_csv(r'C:\Users\VICTUS\Desktop\mastering git\Practise git\healthcare_stroke_predictor\stroke-data.csv')
dataset  = dataset[dataset['gender'] != 'Other']
dataset['bmi'].fillna(dataset['bmi'].mean(),inplace=True)


#Encoding 
le = LabelEncoder()
col_name = ['gender','Residence_type','ever_married']
for col in col_name:
    dataset[col] = le.fit_transform(dataset[col])
    
    
#OneHotencode 
ct = ColumnTransformer(
    transformers=[('encoder',OneHotEncoder(drop='first'),["work_type","smoking_status"])],
    remainder='passthrough'
)

X = ct.fit_transform(dataset.drop(['id','stroke'],axis=1))
y = dataset['stroke'].values

#Train-test-split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

sm = SMOTE(random_state=42)
X_train,y_train = sm.fit_resample(X_train,y_train)

#Standardize
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000,class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "Naive Bayes": GaussianNB()    
}

#Train and Evaluate
results = []

for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred,)
    prec = precision_score(y_test,y_pred,zero_division=0)
    rec = recall_score(y_test,y_pred,zero_division=0)
    f1 = f1_score(y_test,y_pred,zero_division=0)
    results.append([name,acc,prec,rec,f1])
    print(f'{name} Done')
    
    
#Trainig ANN model separately
ann = Sequential([
    Dense(units=8, activation="relu", input_dim=X_train.shape[1]),
    Dense(units=8, activation="relu"),
    Dense(units=1, activation="sigmoid")
])

ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
ann.fit(X_train,y_train,batch_size=32,epochs=50,verbose=0)

y_pred_ann = (ann.predict(X_test)>0.5).astype('int32')

acc = accuracy_score(y_test, y_pred_ann)
prec = precision_score(y_test, y_pred_ann,zero_division=0)
rec = recall_score(y_test, y_pred_ann,zero_division=0)
f1 = f1_score(y_test, y_pred_ann,zero_division=0)

results.append(['ANN',acc,prec,rec,f1])  

# Display Comparison
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
results_df = results_df.sort_values(by="Recall", ascending=False)
print("\n Model Performance Comparison:")
print(results_df)

# Save the Best Model (by Recall)
best_model_name = results_df.iloc[0]["Model"]
if best_model_name == "ANN":
    ann.save("best_model.h5")
    best_model = ann
else:
    best_model = models[best_model_name]
    with open("best_model.pkl", "wb") as f:
        pickle.dump(models[best_model_name], f)

# Save preprocessors
with open("column_transformer.pkl", "wb") as f:
    pickle.dump(ct, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(sc, f)

print(f"\n Best Model based on Recall: {best_model_name}")
print(" Models and transformers saved successfully!")

# Optional: Detailed Report
if best_model_name == "ANN":
    y_pred_best = y_pred_ann
else:
    y_pred_best = best_model.predict(X_test)

print("\n Classification Report:")
print(classification_report(y_test, y_pred_best))

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))



