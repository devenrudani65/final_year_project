import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("cbc_dataset_1000_rows.csv")

# ---------------- USE ALL 12 FEATURES ----------------
X = df[[
    "hemoglobin",
    "wbc",
    "rbc",
    "platelets",
    "mcv",
    "mch",
    "mchc",
    "neutrophils",
    "lymphocytes",
    "monocytes",
    "eosinophils",
    "basophils"
]]

y = df["disease"]
# ------------------------------------------------------

# Encode disease labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=21
)

# ---------------- Logistic Regression ----------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
pickle.dump(lr, open("models/model_lr.pkl", "wb"))

# ---------------- Random Forest ----------------
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
pickle.dump(rf, open("models/model_rf.pkl", "wb"))

# ---------------- Artificial Neural Network ----------------
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

ann = Sequential()
ann.add(Dense(64, activation='relu', input_shape=(12,)))
ann.add(Dense(32, activation='relu'))
ann.add(Dense(len(np.unique(y_encoded)), activation='softmax'))
ann.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ann.fit(X_train, y_train_cat, epochs=30, batch_size=16, verbose=0)

_, ann_acc = ann.evaluate(X_test, y_test_cat, verbose=0)
ann.save("models/model_ann.h5")

# Save scaler and encoder
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))

print("\nTraining Complete ✅")
print("Logistic Regression Accuracy:", lr_acc)
print("Random Forest Accuracy:", rf_acc)
print("ANN Accuracy:", ann_acc)