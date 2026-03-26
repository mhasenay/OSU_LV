import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import sklearn . linear_model as lm
from sklearn . metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

df = pd.read_csv("../LV3/data_C02_emission.csv")

#a) Odaberite željene numeričke veličine s listom naziva stupaca. Podijeliti podatke
#na skup za učenje i skup za testiranje u omjeru 70%-30%
input = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 
             'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']
output = ['CO2 Emissions (g/km)']

X = df[input]
y = df[output]
#print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
'''
#b) Napravit dijagrame raspršenja i prikazati ovisnost emisije CO2 plinova o
#ostalim numeričkim veličinama
for i in range(len(input)):
    plt.figure()
    
    plt.scatter(X_train.iloc[:, i], y_train, color="blue", alpha=0.5, label="Učenje")
    plt.scatter(X_test.iloc[:, i],  y_test,  color="red",  alpha=0.5, label="Test")
    
    plt.xlabel(X_train.columns[i])
    plt.ylabel(output)
    plt.title(f"CO2 Emission (g/km) vs {X_train.columns[i]}")
    plt.legend()
    
    plt.show()
'''
#c) Izvršiti standardizaciju ulaznih veličina na skupu za učenje
#prikazati histograme ulaznih veličina prije i nakon skaliranja
sc = MinMaxScaler()
X_train_n = sc.fit_transform ( X_train )
X_test_n = sc.transform ( X_test )

'''
plt.figure()
for i in range(len(input)):
    plt.subplot(1, 2, 1)
    plt.hist(X_train.iloc[:, i], bins=20, color="blue")
    plt.title("Prije skaliranja")

    plt.subplot(1, 2, 2)
    plt.hist(X_train_n[:, i], bins=20, color="magenta")
    plt.title("Nakon skaliranja")
    plt.show()
'''
#d) Izgradite linearni regresijski model i ispisati u terminal dobivene parametre modela
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n , y_train)
print(f"Parametri modela: {linearModel.coef_}")

#e) Izvršite procjenu izlazne veličine na temelju ulaznih veličina skupa za testiranje
#prikazati pomoću dijagrama raspršenja odnos između stvarnih vrijednosti izlaza i procjene
y_test_p = linearModel.predict(X_test_n)

plt.figure()
plt.scatter(y_test, y_test_p, alpha=0.5, c="green")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Procijenjene vrijednosti")
plt.title("Stvarno vs Procijenjeno")
plt.show()

#f) Izvršite vrednovanje modela na način da se izračunaju vrijednosti regresijskih metrika
#na skupu podataka za testiranje
print("\nMetrike na testnom skupu:")
print(f"MAE = {mean_absolute_error(y_test, y_test_p):.4f}")
print(f"MSE = {mean_squared_error(y_test, y_test_p):.4f}")
print(f"RMSE = {np.sqrt(mean_squared_error(y_test, y_test_p)):.4f}")
print(f"R2 = {r2_score(y_test, y_test_p):.4f}")
print(f"MAPE = {mean_absolute_percentage_error(y_test, y_test_p):.4f}")
