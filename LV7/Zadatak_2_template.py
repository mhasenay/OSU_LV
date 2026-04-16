import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
'''
# ucitaj sliku
img = Image.imread("LV7/imgs/test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

unique_colors = np.unique(img_array, axis=0)
print("Broj različitih boja:", unique_colors.shape[0])
#Broj različitih boja na originalnoj slici je 97924

km = KMeans(n_clusters=5, random_state=0)
km.fit(img_array)

centers = km.cluster_centers_
labels = km.labels_

img_array_aprox = centers[labels]

img_aprox = np.reshape(img_array_aprox, (w,h,d))

# prikaz svakog klastera kao binarne slike
for i in range(km.n_clusters):
     cluster_mask = (labels == i)
    
     # reshape u dimenzije slike
     binary_img = cluster_mask.reshape(w, h)
    
     plt.figure()
     plt.title(f"Cluster {i}")
     plt.imshow(binary_img, cmap='gray')
     plt.axis('off')
     plt.show()

plt.figure()
plt.title(f"K-means slika")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()
#povečanjem broja clustera odnosno broja k dobivamo sliku koja je sve sličnija orignalnoj

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(img_array)
    
    distortions.append(sum(np.min(cdist(img_array, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / img_array.shape[0])
    
    inertias.append(kmeanModel.inertia_)
    
    mapping1[k] = distortions[-1]
    mapping2[k] = inertias[-1]


print("Distortion values:")
for key, val in mapping1.items():
    print(f'{key} : {val}')

plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()
'''

'''
# ucitaj sliku
img = Image.imread("LV7/imgs/test_2.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

unique_colors = np.unique(img_array, axis=0)
print("Broj različitih boja:", unique_colors.shape[0])
#broj različitih boja na originalnoj slici je 14210

km = KMeans(n_clusters=5, random_state=0)
km.fit(img_array)

centers = km.cluster_centers_
labels = km.labels_

img_array_aprox = centers[labels]

img_aprox = np.reshape(img_array_aprox, (w, h, d))
  
    # prikaz svakog klastera kao binarne slike
for i in range(km.n_clusters):
    cluster_mask = (labels == i)
    
     # reshape u dimenzije slike
    binary_img = cluster_mask.reshape(w, h)
    
    plt.figure()
    plt.title(f"Cluster {i}")
    plt.imshow(binary_img, cmap='gray')
    plt.axis('off')
    plt.show()
 
plt.figure()
plt.title(f"K-means slika")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(img_array)
    
    distortions.append(sum(np.min(cdist(img_array, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / img_array.shape[0])
    
    inertias.append(kmeanModel.inertia_)
    
    mapping1[k] = distortions[-1]
    mapping2[k] = inertias[-1]

print("Distortion values:")
for key, val in mapping1.items():
    print(f'{key} : {val}')

plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()
'''
'''
# ucitaj sliku
img = Image.imread("LV7/imgs/test_3.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

unique_colors = np.unique(img_array, axis=0)
print("Broj različitih boja:", unique_colors.shape[0])
#Broj različitih boja na originalnoj slici je 97473

km = KMeans(n_clusters=5, random_state=0)
km.fit(img_array)

centers = km.cluster_centers_
labels = km.labels_

img_array_aprox = centers[labels]

img_aprox = np.reshape(img_array_aprox, (w, h, d))

# # prikaz svakog klastera kao binarne slike
# for i in range(km.n_clusters):
#     cluster_mask = (labels == i)
    
#     # reshape u dimenzije slike
#     binary_img = cluster_mask.reshape(w, h)
    
#     plt.figure()
#     plt.title(f"Klaster {i}")
#     plt.imshow(binary_img, cmap='gray')
#     plt.axis('off')
#     plt.show()

plt.figure()
plt.title(f"K-means slika")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(img_array)
    
    distortions.append(sum(np.min(cdist(img_array, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / img_array.shape[0])
    
    inertias.append(kmeanModel.inertia_)
    
    mapping1[k] = distortions[-1]
    mapping2[k] = inertias[-1]


print("Distortion values:")
for key, val in mapping1.items():
    print(f'{key} : {val}')

plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()
'''
'''
# ucitaj sliku
img = Image.imread("LV7/imgs/test_4.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

unique_colors = np.unique(img_array, axis=0)
print("Broj različitih boja:", unique_colors.shape[0])
#broj različitih boja na originalnoj slici je 14210

km = KMeans(n_clusters=5, random_state=0)
km.fit(img_array)

centers = km.cluster_centers_
labels = km.labels_

img_array_aprox = centers[labels]

img_aprox = np.reshape(img_array_aprox, (w, h, d))

# # prikaz svakog klastera kao binarne slike
# for i in range(km.n_clusters):
#     cluster_mask = (labels == i)
    
#     # reshape u dimenzije slike
#     binary_img = cluster_mask.reshape(w, h)
    
#     plt.figure()
#     plt.title(f"Klaster {i}")
#     plt.imshow(binary_img, cmap='gray')
#     plt.axis('off')
#     plt.show()

plt.figure()
plt.title(f"K-means slika")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(img_array)
    
    distortions.append(sum(np.min(cdist(img_array, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / img_array.shape[0])
    
    inertias.append(kmeanModel.inertia_)
    
    mapping1[k] = distortions[-1]
    mapping2[k] = inertias[-1]

print("Distortion values:")
for key, val in mapping1.items():
    print(f'{key} : {val}')

plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()
'''
'''
# ucitaj sliku
img = Image.imread("LV7/imgs/test_5.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

unique_colors = np.unique(img_array, axis=0)
print("Broj različitih boja:", unique_colors.shape[0])
#Broj različitih boja na originalnoj slici je 151202

km = KMeans(n_clusters=5, random_state=0)
km.fit(img_array)

centers = km.cluster_centers_
labels = km.labels_

img_array_aprox = centers[labels]

img_aprox = np.reshape(img_array_aprox, (w, h, d))

# # prikaz svakog klastera kao binarne slike
# for i in range(km.n_clusters):
#     cluster_mask = (labels == i)
    
#     # reshape u dimenzije slike
#     binary_img = cluster_mask.reshape(w, h)
    
#     plt.figure()
#     plt.title(f"Klaster {i}")
#     plt.imshow(binary_img, cmap='gray')
#     plt.axis('off')
#     plt.show()

plt.figure()
plt.title(f"K-means slika")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(img_array)
    
    distortions.append(sum(np.min(cdist(img_array, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / img_array.shape[0])
    
    inertias.append(kmeanModel.inertia_)
    
    mapping1[k] = distortions[-1]
    mapping2[k] = inertias[-1]


print("Distortion values:")
for key, val in mapping1.items():
    print(f'{key} : {val}')

plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()
'''
'''
# ucitaj sliku
img = Image.imread("LV7/imgs/test_6.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

unique_colors = np.unique(img_array, axis=0)
print("Broj različitih boja:", unique_colors.shape[0])
#Broj različitih boja na originalnoj slici je 164291

km = KMeans(n_clusters=5, random_state=0)
km.fit(img_array)

centers = km.cluster_centers_
labels = km.labels_

img_array_aprox = centers[labels]

img_aprox = np.reshape(img_array_aprox, (w, h, d))

# # prikaz svakog klastera kao binarne slike
# for i in range(km.n_clusters):
#     cluster_mask = (labels == i)
    
#     # reshape u dimenzije slike
#     binary_img = cluster_mask.reshape(w, h)
    
#     plt.figure()
#     plt.title(f"Klaster {i}")
#     plt.imshow(binary_img, cmap='gray')
#     plt.axis('off')
#     plt.show()

plt.figure()
plt.title(f"K-means slika")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(img_array)
    
    distortions.append(sum(np.min(cdist(img_array, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / img_array.shape[0])
    
    inertias.append(kmeanModel.inertia_)
    
    mapping1[k] = distortions[-1]
    mapping2[k] = inertias[-1]

print("Distortion values:")
for key, val in mapping1.items():
    print(f'{key} : {val}')

plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()
'''