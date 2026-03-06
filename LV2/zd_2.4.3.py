import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("LV2/road.jpg")
img = img[:, :, 0].copy()

#a) posvijetli sliku
brightness = 50
brighter_img = np.clip(img.astype(np.uint16)+brightness, 0, 255).astype(np.uint8)
#brighter_img[brighter_img > 255] = 255
#brighter_img = brighter_img.astype(np.uint16)
#brighter_img = np.clip(brighter_img, 0, 255)
#brighter_img = brighter_img.astype(np.uint8)
plt.title('a) Posvijetli sliku')
plt.axis('off')
plt.imshow(brighter_img, cmap="gray")
plt.show()

#b)prikazati samo drugu četvrtinu slike po širini
height, width = img.shape
second_quarter = img[:, width//4 : width//2]
plt.title('b) Druga četvrtina po širini')
plt.axis('off')
plt.imshow(second_quarter, cmap = "gray")
plt.show()

#c)zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu
rotated_img = np.rot90(img, -1)
plt.title('c) Zarotirana slika')
plt.axis('off')
plt.imshow(rotated_img, cmap="gray")
plt.show()

#d)zrcaliti sliku
mirrored_img = img[:, ::-1]
plt.title('d) Zrcaljena slika')
plt.axis('off')
plt.imshow(mirrored_img, cmap = "gray")
plt.show()