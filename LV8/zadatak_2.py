import numpy as np
from tensorflow import keras # type: ignore
import matplotlib.pyplot as plt

model = keras.models.load_model("LV8/mnist_model.keras")

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# skaliranje slike na raspon [0,1]
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_test_s = np.expand_dims(x_test_s, -1)

y_pred_probabilities = model.predict(x_test_s)
y_pred = np.argmax(y_pred_probabilities, axis=1)
y_true = y_test

# indexi krivo klasificiranih slika
misclassified_idx = np.where( y_pred != y_true)[0]
print(f"Broj pogrešno klasificiranih: {len(misclassified_idx)} / {len(y_true)}")

plt.figure(figsize=(8,8))

for i, idx in enumerate(misclassified_idx[:9]):
    plt.subplot(3,3, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Stvarno: {y_true[idx]}  |  Predviđeno: {y_pred[idx]}", fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.show()