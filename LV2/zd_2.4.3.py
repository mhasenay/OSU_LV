import numpy as np
import matplotlib.pyplot as plt

 #uzmi sve redove i stupce i uzmi jedan kanal (crveni), 
 #napravi kopiju slike jer bi u suprotnom mogli utjecati na original
img = plt.imread("LV2/road.jpg")
img = img[:, :, 0].copy()

"""
#a) posvijetli sliku
 #svakom pikselu dodaje se 50, ali ako piksel izlazi iz raspona:
brigter_img = img+50
#ako je < 0 postavi 0, ako je > 255 postavi 255, inače ostavi kako je
brigter_img = np.clip(brigter_img, 0, 255)
plt.imshow(brigter_img, cmap="gray")
plt.show()


#b)prikazati samo drugu četvrtinu slike po širini
height, width = img.shape
second_quarter = img[:, width//4 : width//2]
plt.imshow(second_quarter, cmap = "gray")
plt.show()

#c)zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu
rotated_img = np.rot90(img, -1)
plt.imshow(rotated_img, cmap="gray")
plt.show()
"""
#d)zrcaliti sliku
mirrored_img = img[:, ::-1] #uzmi sve redove, stupce uzimaj unazad
plt.imshow(mirrored_img, cmap = "gray")
plt.show()