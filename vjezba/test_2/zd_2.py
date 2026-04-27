import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

#0 mrtvi; 1 živi
'''
zd.1 Datoteka titanic.csv sadrži podatke o putnicima broda Titanic. 
Upoznajte se sa sljedećim data setom i dodajte programski kod u skriptu pomoću
kojeg možete odgovoriti na sljedeća pitanja:
'''
data = pd.read_csv("test_2/titanic.csv")
data.dropna()
#PassengerId,
#Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

input = ['Pclass', 'Sex', 'Fare', 'Embarked'] #sex prebacit u 0 1, embarked u brojevno isto
output = ['Survived']

X = data[input]
y = data[output]
'''
Skalirajte podatke, podijelit 70:30 i onda ić dalje
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10, stratify=y)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_enc = ohe.fit_transform(X_train[['Sex', 'Embarked']])
X_test_enc = ohe.transform(X_test[['Sex', 'Embarked']])

X_train_enc = pd.DataFrame(X_train_enc, columns=ohe.get_feature_names_out(['Sex', 'Embarked']), index=X_train.index)
X_test_enc = pd.DataFrame(X_test_enc, columns=ohe.get_feature_names_out(['Sex', 'Embarked']), index=X_test.index)

X_train_n = pd.concat([X_train[['Pclass', 'Fare']], X_train_enc], axis = 1)
X_test_n = pd.concat([X_test[['Pclass', 'Fare']], X_test_enc], axis = 1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train_enc)
X_test = sc.transform(X_test_enc)

'''
a) Izradite algoritam KNN na skupu podataka za učenje (uz K=5). Vizualizirajte podatkovne primjere i granicu
odluke
'''
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_train_p = knn_model.predict(X_train)
y_test_p = knn_model.predict(X_test)
#Vizualizacija nije moguća jer imamo 4 ulazne varijable

'''
b) Izračunajte točnost klasifikacije na skupu podataka za učenje i skupu podataka za testiranje. Komentirajte
'''
acc_train_knn = accuracy_score(y_train, y_train_p)
acc_test_knn = accuracy_score(y_test, y_test_p)
print(f"Tocnost KNN(skup za ucenje K:5): {acc_train_knn:.3f}")
print(f"Tocnost KNN(skup za testiranje K:5): {acc_test_knn:.3f}")

'''
c) Pomoću unakrsne validacije odredite optimalnu vrijednost hiperparametara K algoritma KNN
'''

knn = KNeighborsClassifier()
param_grids = {
    "n_neighbors" : range(1,31)
}

grid_search = GridSearchCV(estimator = knn, param_grid= param_grids, cv = 5, scoring= "accuracy")
grid_search.fit(X_train, y_train)

print(f"Najbolji hiperparametar: {grid_search.best_params_['n_neighbors']}")
print(f"Najveca tocnost unakrsne validacije : {grid_search.best_score_:.3f}")

'''
d) Izračunajte točnost klasifikacije na skupu podataka za učenje i skupu podataka za testiranje za dobiveni K
Usporedite dobivene rezultate s rezultatima kada je K=5
'''
knn_model = KNeighborsClassifier(n_neighbors=9)
knn_model.fit(X_train, y_train)

y_pred_train_KNN = knn_model.predict(X_train)
y_pred_test_KNN = knn_model.predict(X_test)

acc_train_knn = accuracy_score(y_train, y_pred_train_KNN)
acc_test_knn = accuracy_score(y_test, y_pred_test_KNN)

print(f"Tocnost KNN(skup za ucenje K:9): {acc_train_knn:.3f}")
print(f"Tocnost KNN(skup za testiranje K:9): {acc_test_knn:.3f}")

#Ista je točnost kao i za K=5