import pandas as pd
import numpy as np

data = pd.read_csv('LV3/data_C02_emission.csv')

'''
a) Koliko mjerenja sadrzi DataFrame? Kojeg tipa je svaka velicina?
Postoje li izostale ili duplicirane vrijednosti? Obrisite ih ako postoje.
Kategoricke velicine pretvorite u tip category
'''
def a_zd():
    print(len(data))
    print(data.info())
    data.drop_duplicates()
    data.dropna()
    #msm da se moraju sve ove koje su str pretvorit u category
    data['Vehicle Class'] = data['Vehicle Class'].astype('category')
    print(data.info())    

#a_zd()

'''
b) Koja tri automobila imaju najveću odnosno najmanju gradsku potrosnju?
Ispisite u terminal: ime proizvodaca, model vozila i kolika je gradska potrosnja
'''
def b_zd():
    least_consumption = data.nsmallest(3, 'Fuel Consumption City (L/100km)')
    print(least_consumption[['Make','Model','Fuel Consumption City (L/100km)']])
    #isto to za most consumption al sa nlargest

#b_zd()

'''
c) Koliko vozila ima velicinu motora izmedu 2.5 i 3.5 L? 
Kolika je prosjecna C02 emisija plinova za ova vozila?
'''
def c_zd():
    between = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
    #uredi ove printove
    print(len(between))
    print(between['CO2 Emissions (g/km)'].mean())

#c_zd()

'''
d) Koliko mjerenja se odnosi na vozila proizvodaca Audi? 
Kolika je prosjecna emisija C02 plinova automobila proizvodaca Audi 
koji imaju 4 cilindara?
'''
def d_zd():
    audi = data[data['Make'] == 'Audi']
    print(len(audi))
    audi_4_cylinders = audi[audi['Cylinders'] == 4]
    print(audi_4_cylinders['CO2 Emissions (g/km)'].mean())

#d_zd()

'''
e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija 
C02 plinova s obzirom na broj cilindara?
'''
def e_zd():
    even_cylinders = data[(data['Cylinders']%2 == 0)]
    print(f"Broj vozila sa 4,6,8... cilindara: {len(even_cylinders)}")
    cylinders = even_cylinders.groupby('Cylinders')
    print(cylinders['CO2 Emissions (g/km)'].mean())

#e_zd()

'''
f) Kolika je prosjecna gradska potrošnja u slucaju vozila koja koriste dizel, 
a kolika za vozila koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
'''
def f_zd():
    diesel = data[data['Fuel Type'] == 'D']
    petrol = data[data['Fuel Type'] == 'Z']

    print(f"Dizeli:\nProsjecno: {diesel['Fuel Consumption City (L/100km)'].mean()} - Medijalno: {diesel['Fuel Consumption City (L/100km)'].median()}")
    #isti print za benzin

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
    print (data.corr( numeric_only = True ))

#i_zd()
'''
Velicine imaju dosta veliki korelaciju. Npr. broj obujam motora i broj cilindara su oko 0.9, dok je potrosnja oko 0.8 sto ukazuje na veliku korelaciju.
Takodjer razlog zasto potrosnja u mpg ima veliku negativnu korelaciju je to sto je ta velicina obrnuta, odnosno, sto automobil vise trosi, broj je manji
Npr: automobil koji trosi 25 MPG trosi vise nego automobil koji trosi 45 MPG. Dakle, ta velicina je obrnuta L/100km te takodjer, zbog toga dobivamo negativnu
korelaciju. Sto je negativna korelacija blize -1 to je ona vise obrnuto proporcijalna, dok sto je blize 1, to je vise proporcijonalna. Vrijednosti oko 0
nemaju nikakvu korelaciju s velicinom.
'''