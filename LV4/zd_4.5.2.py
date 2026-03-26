import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, max_error

df = pd.read_csv("../LV3/data_C02_emission.csv")

input = ["Engine Size (L)", "Cylinders", "Fuel Consumption Comb (L/100km)", "Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Type"]
output = "CO2 Emissions (g/km)"

#Na temelju prošlog zadatka izraditi model koji koristi Fuel Type kao ulaznu veličinu
#Koristiti OneHotEncoder
#Ne skalirati ulazne podatke
X = df[input]
y = df[output]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
ohe = OneHotEncoder ()
X_encoded_train = ohe.fit_transform(X_train[['Fuel Type']]).toarray()
X_encoded_test = ohe.fit_transform(X_test[['Fuel Type']]).toarray()

linearModel = lm.LinearRegression ()
linearModel.fit(X_encoded_train, y_train )

y_test_p = linearModel.predict(X_encoded_test)

'''
print("\nMetrike na testnom skupu:")
print(f"MAE  = {mean_absolute_error(y_test, y_test_p):.4f}")
print(f"MSE  = {mean_squared_error(y_test, y_test_p):.4f}")
print(f"RMSE = {np.sqrt(mean_squared_error(y_test, y_test_p)):.4f}")
print(f"R2   = {r2_score(y_test, y_test_p):.4f}")
print(f"MAPE = {mean_absolute_percentage_error(y_test, y_test_p):.4f}")
'''
print("Maksimalna pogreška:", max_error(y_test, y_test_p))

errors = np.abs(y_test.values - y_test_p)
max_error_id = np.argmax(errors)

max_error_model = df.iloc[max_error_id, 1]
print(f"Model vozila s najvećom pogreškom: {max_error_model}")

'''
plt.figure()
plt.scatter(y_test, y_test_p, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Procijenjene vrijednosti")
plt.title("Stvarno vs Procijenjeno")
plt.show()
'''