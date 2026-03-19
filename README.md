# my_ml_project
# Insurance Charges Prediction using Linear Regression

## 📌 Project Overview

This project predicts medical insurance charges using a **Linear Regression model**. The dataset used contains features like age, sex, BMI, number of children, smoking status, and region.

The goal is to train a machine learning model that can estimate insurance costs based on these input features.

---

## 📂 Dataset

Dataset used: `insurance.csv`

### Features:

* `age` → Age of the person
* `sex` → Gender (male/female)
* `bmi` → Body Mass Index
* `children` → Number of children
* `smoker` → Smoking status (yes/no)
* `region` → Residential area
* `charges` → Medical insurance cost (Target Variable)

---

## ⚙️ Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🔄 Workflow

1. Load dataset using Pandas
2. Data preprocessing:

   * Drop unnecessary column (`region`)
   * Convert categorical values (sex → numeric)
3. Split data into training and testing sets
4. Train model using Linear Regression
5. Predict values on test data
6. Evaluate model using R² score

---

## 🧠 Model Used

**Linear Regression**

It tries to find a linear relationship between input features and target variable (charges).

---

## 📊 Evaluation Metric

* **R² Score (R-squared)**

### Formula:

R² = 1 - (SS_res / SS_tot)

* Value ranges from 0 to 1
* Higher value → better model performance

---

## 💻 Code Explanation

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load dataset
df = pd.read_csv("insurance.csv")

# Feature selection
X = df.drop(columns=["charges", "region"])
y = df["charges"]

# Encoding
X["sex"] = X["sex"].map({"male": 0, "female": 1})

from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score

# Evaluation
r2 = r2_score(y_test, y_pred)

print("R squared value is", r2)
print("Accuracy is:", r2 * 100)
```

---

## 📈 Output

* R² Score: Shows how well the model fits the data
* Accuracy: R² × 100

Example:

```
R squared value is 0.75
Accuracy is: 75%
```

---

## ⚠️ Limitations

* Linear Regression assumes linear relationship
* Does not handle complex patterns well
* Missing encoding for other categorical features like `smoker`

---

## 🚀 Future Improvements

* Use One-Hot Encoding for categorical variables
* Try advanced models (Random Forest, XGBoost)
* Hyperparameter tuning
* Feature scaling

---

## 👨‍💻 Author

Sachin Kumar

---

## 📌 Conclusion

This project demonstrates a basic machine learning pipeline for predicting insurance charges. It is a good starting point for beginners to understand regression models.
