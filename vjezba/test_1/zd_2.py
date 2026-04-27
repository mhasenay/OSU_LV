import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import sklearn.linear_model as lm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score


column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
data = pd.read_csv('test_1/pima-indians-diabetes.csv', skiprows=9, header = None, names = column_names)
data_cleaned = data.dropna(subset=['Age', 'BMI'])
data_cleaned = data_cleaned[(data_cleaned['BMI'] != 0)]

'''
Zd.2 Podijeliti data na ulazne podatke X i izlazne podatke y. Podijeliti podatke na skup za učenje i skup za
testiranje modela u omjeru 80:20. Dodajte programski kod u skriptu pomoću kojeg možete odgovoriti na sljedeća pitanja
'''
input = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age']

output = ['Outcome']

X = data_cleaned[input]
y = data_cleaned[output]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

'''
a) Izgraditi model logističke regresije pomoću scikit-learn biblioteke na temelju skupa podataka za učenje
'''
model = LogisticRegression(max_iter=1200)
model.fit(X_train, y_train)

'''
b) Provedite klasifikaciju skupa podataka za testiranje pomoću izgrađenog modela
'''
y_test_p = model.predict(X_test)

'''
c) Izračunajte i prikažite matricu zabune na testnim podatcima. Komentirajte dobivene rezultate
'''
disp = ConfusionMatrixDisplay( confusion_matrix(y_test , y_test_p ))
disp.plot(cmap = 'PuRd')
plt.title('Matrica zabune na testnim podatcima')
plt.show()

'''
d) Izračunajte točnost, preciznost i odziv na skupu podataka za testiranje. Komentirajte dobivene rezultate
'''
print("---Evaluacija modela---")
print(f"Točnost: {accuracy_score(y_test, y_test_p)}")
print(f"Preciznost: {precision_score(y_test, y_test_p)}")
print(f"Odziv: {recall_score(y_test, y_test_p)}")