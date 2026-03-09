import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('LV3/data_C02_emission.csv')

'''
a) Koliko mjerenja sadrzi DataFrame? Kojeg tipa je svaka velicina?
Postoje li izostale ili duplicirane vrijednosti? Obrisite ih ako postoje.
Kategoricke velicine pretvorite u tip category
'''
def a_zd():
    #print(len(data))
    print(data.info())
    print(data.isnull().sum())
    data.dropna()
    data.drop_duplicates()
    data['Make'] = data['Make'].astype('category')
    data['Model'] = data['Model'].astype('category')
    data['Vehicle Class'] = data['Vehicle Class'].astype('category')
    data['Transmission'] = data['Transmission'].astype('category')
    data['Fuel Type'] = data['Fuel Type'].astype('category')

    print(data.info())    

#a_zd()

'''
b) Koja tri automobila imaju najveću odnosno najmanju gradsku potrosnju?
Ispisite u terminal: ime proizvodaca, model vozila i kolika je gradska potrosnja
'''
def b_zd():
    least_consumption = data.nsmallest(3, 'Fuel Consumption City (L/100km)')
    print("3 cars with least consumption: ")
    print(least_consumption[['Make','Model','Fuel Consumption City (L/100km)']])
    most_consumption = data.nlargest(3, 'Fuel Consumption City (L/100km)')
    print("3 cars with most consumption: ")
    print(most_consumption[['Make','Model','Fuel Consumption City (L/100km)']])

#b_zd()

'''
c) Koliko vozila ima velicinu motora izmedu 2.5 i 3.5 L? 
Kolika je prosjecna C02 emisija plinova za ova vozila?
'''
def c_zd():
    between_engines = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
    print(f"Broj vozila s veličinom motora između 2.5 i 3.5 L: {len(between_engines)}")
    print(f"Prosjecna CO2 emisija plinova: {between_engines['CO2 Emissions (g/km)'].mean():.4f}")

#c_zd()

'''
d) Koliko mjerenja se odnosi na vozila proizvodaca Audi? 
Kolika je prosjecna emisija C02 plinova automobila proizvodaca Audi 
koji imaju 4 cilindara?
'''
def d_zd():
    audis = data[data['Make'] == 'Audi']
    print(f"Broj Audija: {len(audis)}")
    audis_4_cylinders = audis[audis['Cylinders'] == 4]
    print(f"Prosječna emisija CO2 Audija s 4 cilindra: {audis_4_cylinders['CO2 Emissions (g/km)'].mean():.4f}")

#d_zd()

'''
e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija 
C02 plinova s obzirom na broj cilindara?
'''
def e_zd():
    even_cylinders = data[(data['Cylinders'] % 2 == 0)]
    print(f"Broj vozila sa 4,6,8... cilindara: {len(even_cylinders)}")
    cylinders = even_cylinders.groupby('Cylinders')
    print(cylinders.size())
    print(cylinders['CO2 Emissions (g/km)'].mean())

#e_zd()

'''
f) Kolika je prosjecna gradska potrošnja u slucaju vozila koja koriste dizel, 
a kolika za vozila koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
'''
def f_zd():
    diesel = data[data['Fuel Type'] == 'D']
    r_gasoline = data[data['Fuel Type'] == 'X']

    print(f"Dizel:\nProsječne vrijednosti: {diesel['Fuel Consumption City (L/100km)'].mean()}\nMedijalne vrijednosti: {diesel['Fuel Consumption City (L/100km)'].median()}")
    print(f"Regularni benzin:\nProsječne vrijednosti: {r_gasoline['Fuel Consumption City (L/100km)'].mean()}\nMedijalne vrijednosti: {r_gasoline['Fuel Consumption City (L/100km)'].median()}")

#f_zd()

'''
g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu 
gradsku potrošnju goriva?
'''
def g_zd():
    diesel_4_cylinders = data[(data['Fuel Type'] == 'D') & (data['Cylinders'] == 4)]
    print(diesel_4_cylinders.nlargest(1, 'Fuel Consumption City (L/100km)'))

#g_zd()

'''
h) Koliko ima vozila ima rucni tip mjenjaca (bez obzira na broj brzina)?
'''
def h_zd():
    manual = data[data['Transmission'].str.startswith('M')]
    print(len(manual))

#h_zd()

'''
i) Izracunajte korelaciju izmedu numerickih velicina. Komentirajte dobiveni rezultat.
'''
def i_zd():
    corr = data.corr( numeric_only = True )
    sns.heatmap(corr, cmap="cool", annot=True)
    plt.show()

#i_zd()