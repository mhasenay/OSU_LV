import numpy as np
from tensorflow import keras
from PIL import Image

model = keras.models.load_model("LV8/mnist_model.keras")

img = Image.open("LV8/test_2.png")

img = img.convert("L") #stavlja u grayscale

img = img.resize((28, 28))

img_array = np.array(img).astype("float32")
img_array = 255 - img_array  # <-- bijela pozadina -> crna, crna znamenka -> bijela
img_array = img_array / 255

# 4. Dodaj dimenziju za batch i kanal: (1, 28, 28, 1)
img_array = np.expand_dims(img_array, axis=(0, -1))

#Klasifikicija
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
confidence = predictions[0][predicted_class] * 100 # ??

print(f"\nPedviđena znamenka: {predicted_class}")
print(f"Pouzdanost: {confidence:.2f}%")

print("\nVjerojatnosti za svaku znamenku:")
for i, prob in enumerate(predictions[0]):
    print(f"  {i}: {prob*100:5.1f}%")

import matplotlib.pyplot as plt
plt.imshow(img_array[0, :, :, 0], cmap='gray')
plt.title("Slika kakvu mreža vidi")
plt.show()