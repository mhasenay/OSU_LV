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
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from keras import models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
podijelit 75:25 i onda ić dalje
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10, stratify=y)

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
a) Izgradite neuronsku mrežu sa sljedećim karakteristikama:
-model očekuje ulazne podatke X
-prvi skriveni sloj ima 12 neurona i koristi relu
-drugi skriveni sloj ima 8 neurona i koristi relu
-treći skriveni sloj ima 4 neurona i koristi relu
-izlazni sloj ima jedan neuron i koristi sigmoid
Ispisati podatke o mreži u terminal
'''
model = keras.Sequential()
model.add(layers.Dense(12, activation='relu', input_shape = (X_train.shape[1],)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

'''
b) Podesiti proces treniranja mreže sa sljedećim parametrima
-loss: binary_crossentropy
-optimizer: adam
-metrika: accuracy
'''
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy",])

'''
c) Pokreniti učenje mreže sa proizvoljnim brojem epoha (pokušat sa 100) i veličinom batcha 5
'''
epoch = 100
batch_size = 5
history = model.fit(X_train,
                    y_train,
                    batch_size = batch_size,
                    epochs = epoch,
                    validation_split = 0.1)

'''
d) Pohraniti model na tvrdi disk te preostale zadatke napravit preko učitanog modela
'''
model.save("test_2/titanic_model.keras")

'''
e) Izvršiti evaluaciju mreže na testnom skupu podataka
'''
model = models.load_model("test_2/titanic_model.keras")
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Testni gubitak (loss): {loss}")
print(f"Testna točnost (accuracy): {accuracy}")

'''
f) Izvršiti predikciju mreže na skupu podataka za testiranje. Prikazati matricu zabune za skup podataka za testiranje.
Komentirati dobivene rezultate i predložiti kako bi ih poboljšali ako je potrebno
'''
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

disp = ConfusionMatrixDisplay( confusion_matrix(y_test , y_pred ))
disp.plot(cmap = 'PuRd')
plt.title('Matrica zabune za testni skup')
plt.show()

#Ovo je značajan napredak od onog modela iz 2. zadatka jer zapravo razlikuje
#žive i mrtve!!!!! 80% točnost je i više nego dobra

'''
Za poboljšanje modela dodala bih stupac Age u model te ga popunila 
medijanima godina jer su djeca imala ogromnu stopu preživljavanja, a model uopće nema 
tu informaciju. Također bih dodala Dropout slojeve i povećala batch size
'''