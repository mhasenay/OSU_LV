import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('LV3/data_C02_emission.csv')

'''
a) Pomocu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz.
'''
def a_zd():
    plt.figure()
    data['CO2 Emissions (g/km)'].plot(kind='hist', bins=20, color = 'yellow', edgecolor = 'black')
    plt.title('Emisija C02 plinova')
    plt.show()

#a_zd()

'''
b)Pomocu dijagrama raspršenja prikažite odnos izmedu gradske potrošnje goriva i emisije
C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izmedu
velicina, obojite tockice na dijagramu raspršenja s obzirom na tip goriva.
'''
def b_zd():
    data.plot.scatter(x = 'Fuel Consumption City (L/100km)', y = 'CO2 Emissions (g/km)', c = 'Fuel Type', colormap = 'coolwarm')
    plt.title('Odnos gradske potrošnje goriva i emisije C02 plinova')
    plt.xlabel('Gradska potrošnja goriva (L/100km)')
    plt.ylabel('Emisija C02 plinova (g/km)')
    plt.show()

#b_zd()

'''
c) Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s 
obzirom na tip goriva. Primjecujete li grubu mjernu pogrešku u podacima?
'''
def c_zd():
    data.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type')
    plt.title('Razdioba izvangradske potrošnje po tipu goriva')
    plt.suptitle('')
    plt.xlabel('Tip goriva')
    plt.ylabel('Izvangradska potrošnja (L/100km)')
    plt.show()

#c_zd()

'''
d) Pomocu stupcastog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu
groupby.
'''
def d_zd():
    fuel_grouped = data.groupby('Fuel Type').size()
    fuel_grouped.plot(kind='bar', xlabel = 'Fuel Type', ylabel = 'Number of vehicles', color = 'cyan', edgecolor = 'black')
    plt.show()

#d_zd()

'''
e) Pomocu stupcastog grafa prikažite na istoj slici prosjecnu C02 emisiju vozila 
s obzirom na broj cilindara.
'''
def e_zd():
    cylinders_grouped = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
    cylinders_grouped.plot(kind='bar', xlabel = 'Number of Cylinders', ylabel = 'Average CO2 Emissions (g/km)', color = 'magenta', edgecolor = 'black')
    plt.show()

#e_zd()