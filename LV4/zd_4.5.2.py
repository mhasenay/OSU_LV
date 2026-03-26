import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, max_error

df = pd.read_csv("data_C02_emission.csv")

input = ["Engine Size (L)", "Cylinders", "Fuel Consumption Comb (L/100km)", "Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Type"]
output = "CO2 Emissions (g/km)"

X = df[input]
y = df[output]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
sc = MinMaxScaler()
X_train_n = sc.fit_transform ( X_train )
X_test_n = sc.transform ( X_test )




linearModel = lm. LinearRegression ()
linearModel . fit ( X_train_n , y_train )


y_test_p = linearModel . predict ( X_test_n )


print("Maksimalna pogreška:", max_error(y_test, y_test_p))


plt.figure()
plt.scatter(y_test, y_test_p, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Procijenjene vrijednosti")
plt.title("Stvarno vs Procijenjeno")
plt.show()


print("\nMetrike na testnom skupu:")
print(f"MAE  = {mean_absolute_error(y_test, y_test_p):.4f}")
print(f"MSE  = {mean_squared_error(y_test, y_test_p):.4f}")
print(f"RMSE = {np.sqrt(mean_squared_error(y_test, y_test_p)):.4f}")
print(f"R2   = {r2_score(y_test, y_test_p):.4f}")
print(f"MAPE = {mean_absolute_percentage_error(y_test, y_test_p):.4f}")
