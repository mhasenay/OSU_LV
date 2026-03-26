import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import sklearn . linear_model as lm
from sklearn . metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

df = pd.read_csv("data_C02_emission.csv")

input = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 
             'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']
output = ['CO2 Emissions (g/km)']

X = df[input]
y = df[output]
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


for i in range(len(input)):
    plt.figure()
    
    plt.scatter(X_train.iloc[:, i], y_train, color="blue", alpha=0.4, label="Učenje")
    plt.scatter(X_test.iloc[:, i],  y_test,  color="red",  alpha=0.4, label="Test")
    
    plt.xlabel(X_train.columns[i])
    plt.ylabel(output)
    plt.title(f"CO2 emisija vs {X_train.columns[i]}")
    plt.legend()
    
    plt.show()


sc = MinMaxScaler()
X_train_n = sc.fit_transform ( X_train )
X_test_n = sc.transform ( X_test )


plt.figure()
for i in range(len(input)):
    plt.subplot(1, 2, 1)
    plt.hist(X_train.iloc[:, i], bins=30, color="blue")
    plt.title("Prije skaliranja")

    plt.subplot(1, 2, 2)
    plt.hist(X_train_n[:, i], bins=30, color="orange")
    plt.title("Nakon skaliranja")
    plt.show()

ohe = OneHotEncoder ()
X_encoded = ohe.fit_transform(df[['Fuel Type']]).toarray()
linearModel = lm. LinearRegression ()
linearModel.fit( X_train_n , y_train )


y_test_p = linearModel . predict ( X_test_n )


plt.figure()
plt.scatter(y_test, y_test_p, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Procijenjene vrijednosti")
plt.title("Stvarno vs Procijenjeno")
plt.show()

print("\nMetrike na testnom skupu:")
print(f"  MAE  = {mean_absolute_error(y_test, y_test_p):.4f}")
print(f"  MSE  = {mean_squared_error(y_test, y_test_p):.4f}")
print(f"  RMSE = {np.sqrt(mean_squared_error(y_test, y_test_p)):.4f}")
print(f"  R2   = {r2_score(y_test, y_test_p):.4f}")
print(f"  MAPE   = {mean_absolute_percentage_error(y_test, y_test_p):.4f}")
