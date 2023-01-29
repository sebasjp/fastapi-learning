from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

pd.options.display.max_columns = None

# prepare data to train
housing = fetch_california_housing()
data = pd.DataFrame(housing["data"], columns=housing["feature_names"])
data[housing["target_names"][0]] = housing["target"]
data.columns = data.columns.str.lower()

data["averooms"] = data["averooms"].round()
data["avebedrms"] = data["avebedrms"].round()

bounds_down = {"avebedrms": 1}
bounds_up = {"averooms": 10, "avebedrms": 2}

for x, val in bounds_down.items():
    data = data[data[x] >= val]

for x, val in bounds_up.items():
    data = data[data[x] <= val]

target_str = "medhouseval"
cols_to_use = ["houseage", "averooms", "avebedrms"]

X = data[cols_to_use].copy()
y = data[target_str].copy().values

# we are going to standardized the variables in order to implement
# this processing in FastAPI, just to practice,
# since, we know that we don't need to, in order to use RF
scaler = StandardScaler()
scaler = scaler.fit(X)
X[X.columns] = scaler.transform(X)

# train model
model = RandomForestRegressor(random_state=421)
model = model.fit(X, y)

# metric
mdpe = np.median( np.abs(y - model.predict(X)) / y)
print(f"MDPE={mdpe}")

# save model and scaler
joblib.dump(scaler, "model_files/scaler.pkl")
joblib.dump(model, "model_files/model.pkl")


