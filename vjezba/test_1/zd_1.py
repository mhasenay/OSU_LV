import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
Zd.1 Datoteka sadrži mjerenja provedena u svrhu otkrivanja dijabetesa, pri čemu se u 9. stupcu
nalazi klasa 0 (nema dijabetes) ili klasa 1 (ima dijabetes). Učitati dane podatke u obliku
numpy polja data. Dodajte programski kod u skriptu pomoću kojega možete odgovoriti na pitanja:
'''
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
data = pd.read_csv('test_1/pima-indians-diabetes.csv', skiprows=9, header = None, names = column_names)

'''
a) Na temelju veličine numpy polja data, na koliko osoba su izvršena mjerenja
'''
numberOfPeople = data.shape[0]
print(f"Broj osoba na kojima su izvršena mjerenja: {numberOfPeople}")
'''
b) Postoje li izostale ili duplicirane vrijednosti u stupcima s mjerenjima dobi i indeksa
tjelesne mase? Obrišite ih ako postoje. Koliko je sad uzoraka ostalo
'''
print("Izostale vrijednosti (NaN):")
print(data[['Age', 'BMI']].isnull().sum())

print("\nBroj nula u 'Age' i 'BMI':")
print((data[['Age', 'BMI']] == 0).sum())

data_cleaned = data.dropna(subset=['Age', 'BMI'])
data_cleaned = data_cleaned[(data_cleaned['BMI'] != 0)]

print(f"\nBroj preostalih uzoraka nakon čišćenja: {len(data_cleaned)}")

'''
c) Prikažite odnos dobi i indeksa tjelesne mase osobe pomoću scatter dijagrama. Dodajte naziv
dijagrama i osi s pripadajućim mjernim jedinicama. Komentirajte odnos dobi i BMI prikazan 
dijagramom
'''

plt.scatter(data_cleaned['Age'], data_cleaned['BMI'], s = 5, c='purple')
plt.title('Odnos dobi i indeksa tjelesne mase (BMI)')
plt.xlabel('Age (years)')
plt.ylabel('BMI ((kg/m)^2)')
plt.show()

'''
d) Izračunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost indeksa
tjelesne mase u ovom podatkovnom skupu
'''

min_bmi = data_cleaned['BMI'].min()
max_bmi = data_cleaned['BMI'].max()
mean_bmi = data_cleaned['BMI'].mean()

print(f"Minimalna: {min_bmi}\nMaksimalna: {max_bmi}\nSrednja: {mean_bmi}")

'''
e) Ponovite zadatak pod d), ali posebno za osobe koje imaju dijabetes i koje nemaju. Koliko ljudi
ima dijabetes? Komentirati...
'''
diabetics = data_cleaned[data_cleaned['Outcome'] == 1]
non_diabetics = data_cleaned[data_cleaned['Outcome'] == 0]

print("Osobe sa dijabetesom:")
print(f"Minimalna vrijednost BMI: {diabetics['BMI'].min()}")
print(f"Maksimalna vrijednost BMI: {diabetics['BMI'].max()}")
print(f"Srednja vrijednost BMI: {diabetics['BMI'].mean()}")
print()

print("Osobe koje nemaju dijabetes:")
print(f"Minimalna vrijednost BMI: {non_diabetics['BMI'].min()}")
print(f"Maksimalna vrijednost BMI: {non_diabetics['BMI'].max()}")
print(f"Srednja vrijednost BMI: {non_diabetics['BMI'].mean()}")
print()

print(f"Dijabetes je dijagnosticiran {diabetics.shape[0]} osoba.")