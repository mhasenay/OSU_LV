import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('LV2/data.csv', delimiter = ',', skip_header = 1)

#a)
broj_osoba = data.shape[0]
print(f"Broj osoba na kojima su izvršena mjerenja: {broj_osoba}")

#b) c)
visina = data[:,1]
masa = data[:,2]

plt.figure(1)
plt.scatter(visina, masa, s = 5, c='purple')
plt.axhline(y=70, color = 'red', linestyle = "--")
plt.title('Odnos visine i mase')
plt.xlabel('Visina (cm)')
plt.ylabel('Masa (kg)')
plt.show()

plt.figure(2)
plt.scatter(visina[::50], masa[::50], c='orange')
plt.title('Odnos visine i mase svake 50. osobe')
plt.xlabel('Visina (cm)')
plt.ylabel('Masa (kg)')

plt.show()

#d)
min_visina = visina.min()
max_visina = visina.max()
srednja_visina = visina.mean()
print(f"Minimalna: {min_visina}cm\nMaksimalna: {max_visina}cm\nSrednja: {srednja_visina}cm")

#e)
muskarci = data[data[:, 0] == 1]
muskarci_visine = muskarci[:,1]
zene = data[data[:,0] == 0]
zene_visine = zene[:,1]

print("Muškarci - min:", muskarci_visine.min())
print("Muškarci - max:", muskarci_visine.max())
print("Muškarci - mean:", muskarci_visine.mean())

print("Žene - min:", zene_visine.min())
print("Žene - max:", zene_visine.max())
print("Žene - mean:", zene_visine.mean())