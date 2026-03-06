import numpy as np
import matplotlib.pyplot as plt

crni = np.zeros((50, 50))
bijeli = np.ones((50, 50))*255

roza = [255, 105, 180]
zuta = [255, 255, 0]

rozi = np.full((50, 50, 3), roza)
zuti = np.full((50, 50, 3), zuta)

gornji_red = np.hstack((rozi, zuti))
donji_red = np.hstack((zuti, rozi))

fin = np.vstack((gornji_red, donji_red))

plt.imshow(fin, cmap="gray", vmin = 0, vmax = 255)
plt.title("Zadatak 2.4.4")
plt.show()