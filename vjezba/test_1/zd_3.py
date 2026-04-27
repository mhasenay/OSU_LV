import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import sklearn.linear_model as lm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score
from tensorflow import keras
from tensorflow.keras import layers

column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
data = pd.read_csv('test_1/pima-indians-diabetes.csv', skiprows=9, header = None, names = column_names)
data_cleaned = data.dropna(subset=['Age', 'BMI'])
data_cleaned = data_cleaned[(data_cleaned['BMI'] != 0)]

'''
zd.3 Podijeliti podatke na X ulazne i y izlazne i podijelit ih na skup za učenje i skup za testiranje
u omjeru 80:20.
'''
input = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age']

output = ['Outcome']

X = data_cleaned[input]
y = data_cleaned[output]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

'''
a) Izgraditi neuronsku mrežu sa sljedećim karakteristikama:
-model očekuje ulazne podatke s 8 varijabli
-prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
-drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju
-izlazni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju
Ispisati podatke o mreži u terminal
'''
print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))

model = keras.Sequential()
model.add(layers.Input(shape = (8,)))
model.add(layers.Dense(12, activation = 'relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

'''
b) Podesite proces treniranja mreže sa sljedećim parametrima:
-loss argument: cross entropy
-optimizer adam
-metrika: accuracy
'''
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy',])

'''
c) Pokreniti učenje s proizvoljnim brojem epoha (otp 150) i veličinom batcha 10
'''
batch_size = 10
epoch = 150
history = model.fit(X_train,
                    y_train,
                    batch_size = batch_size,
                    epochs = epoch,
                    validation_split = 0.1)

'''
d) Pohranite model na tvrdi disk te preostale zadatke riješite koristeći pohranjeni model
'''
model.save("test_1/model.keras")

'''
e)Izvršite evaluaciju mreže na testnom skupu podataka
'''
model = keras.models.load_model("test_1/model.keras")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

'''
f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
podataka za testiranje. Komentirajte dobivene rezultate
'''
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

disp = ConfusionMatrixDisplay( confusion_matrix(y_test , y_pred ))
disp.plot(cmap = 'PuRd')
plt.title('Matrica zabune za testni skup')
plt.show()

#Model točno predviđa 99 primjera klase 0 odnosno nema dijabetes, a pogrešno je predvidio 53 primjera 
#koji su zapravo klasa 1. Model uopće ne predviđa klasu 1 odnosno ima dijabetes
#Točnost iznosi oko 65% što na prvu nije loše, ali zavarava jer model pogađa isključivo jer je klasa 0
#brojnija u skupu podataka

#Model radi linijom manjeg otpora gdje mu je lakše uvijek predvidjeti klasu 0 bez razmišljanja 