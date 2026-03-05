import numpy as np
import matplotlib.pyplot as plt

crni = np.zeros((50, 50))
bijeli = np.ones((50, 50))*255

gornji_red = np.hstack((crni, bijeli))
donji_red = np.hstack((bijeli, crni))

fin = np.vstack((gornji_red, donji_red))

plt.imshow(fin, cmap="gray", vmin = 0, vmax = 255)
plt.title("Zadatak 2.4.4")
plt.show()