import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#0 mrtvi; 1 živi
'''
zd.1 Datoteka titanic.csv sadrži podatke o putnicima broda Titanic. 
Upoznajte se sa sljedećim data setom i dodajte programski kod u skriptu pomoću
kojeg možete odgovoriti na sljedeća pitanja:
'''
data = pd.read_csv("test_2/titanic.csv")

#PassengerId,
#Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

'''
a) Za koliko žena postoje podatci u ovom skupu podataka
'''
females = data[data['Sex'] == 'female']
print(f"Broj žena: {females.shape[0]}")

'''
b) Koliki postotak osoba nije preživio potonuće broda
'''
number_of_people = data.shape[0]
number_of_deaths = data[data['Survived'] == 0].shape[0]
print(f"Postotak mrtvih: {((number_of_deaths/number_of_people)*100):.2f}%")

'''
c) Pomoću stupčastog dijagrama prikažite postotke preživjelih muškaraca (zelenom) i žena (žutom)
Dodajte naziv osi i naziv dijagrama. Komentirajte korelaciju spola i postotka preživljavanja
'''

males = data[data['Sex'] == 'male']
survived_males_pct = ((males[males['Survived'] == 1].shape[0])/males.shape[0])*100
survived_females_pct = ((females[females['Survived'] == 1].shape[0])/females.shape[0])*100

labels = ['Muškarci', 'Žene']
pct = [survived_males_pct, survived_females_pct]
colors = ['g', 'y']
'''
plt.bar(labels, pct, color=colors)
plt.title('Postotak preživjelih po spolu')
plt.xlabel('Spol')
plt.ylabel('Postotak')
plt.grid(axis = 'y', alpha = 0.7, color='b')
plt.show()
'''
'''
d) Kolika je prosječna dob svih preživjelih žena, a koliko muškaraca
'''
survived_males = males[males['Survived'] == 1]
survived_females = females[females['Survived'] == 1]

print(f"Prosječna dob preživjelih žena: {survived_females['Age'].mean():.2f}\nProsječna dob preživjelih muškaraca: {survived_males['Age'].mean():.2f}")

'''
e) Koliko godina ima najstariji preživjeli muškarac u svakoj od klasa? Komentirajte
'''

oldest_men = survived_males.groupby('Pclass')['Age'].max()
print(oldest_men)