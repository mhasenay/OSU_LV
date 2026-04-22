import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
plt.figure(figsize=(6,6))

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')

plt.show()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape = (28,28,1)))
model.add(layers.Flatten()) #Dense treba vektore
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(10, activation="softmax")) #izlazi su vjerojatnosti koje kad se zbroje su 1


model.summary()


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy",])


# TODO: provedi ucenje mreze
batch_size=32
epoch = 20
history = model.fit(x_train_s,
                    y_train_s,
                    batch_size = batch_size,
                    epochs = epoch,
                    validation_split = 0.1)

predictions = model.predict(x_test_s)


# TODO: Prikazi test accuracy i matricu zabune
print(x_test_s.shape)

test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

y_pred_probabilities = model.predict(x_test_s)
y_pred = np.argmax(y_pred_probabilities, axis=1)
y_true = y_test

cm = confusion_matrix(y_true, y_pred)
print(cm)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.show()


# TODO: spremi model
model.save("models\mnist_model.keras")