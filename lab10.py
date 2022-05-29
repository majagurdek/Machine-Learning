# -*- coding: utf-8 -*-
"""lab10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CaqSSa0VoW9vktThVHlArfoXTW0-GTtW
"""

import numpy as np
import pandas as pd

import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data() 
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

X_train = tf.keras.layers.Rescaling(scale=1./255)(X_train)
X_test = tf.keras.layers.Rescaling(scale=1./255)(X_test)

import matplotlib.pyplot as plt 
plt.imshow(X_train[142], cmap="binary") 
plt.axis('off')
plt.show()

class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka","sandał", "koszula", "but", "torba", "kozak"]
class_names[y_train[142]]

from tensorflow import keras


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
#model.add(keras.layers.Dense(300))
#model.add(keras.layers.Dense(100))
#model.add(keras.layers.Dense(10))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))

model.summary()
tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)

model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd", metrics=["accuracy"])

import os
root_logdir = os.path.join(os.curdir, "image_logs")
def get_run_logdir(): 
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") 
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

model.fit(X_train, y_train, epochs=20, validation_split=0.1, callbacks=[tensorboard_cb])

image_index = np.random.randint(len(X_test))
image = np.array([X_test[image_index]])
confidences = model.predict(image)
confidence = np.max(confidences[0])
prediction = np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir ./image_logs

model.save('fashion_clf.h5')

#regresja

from sklearn.datasets import fetch_california_housing 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data,housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

model2 = keras.models.Sequential()
model2.add(keras.layers.Dense(30, activation="relu"))
model2.add(keras.layers.Dense(1))

model2.compile(loss="mean_squared_error",optimizer="sgd")

es = tf.keras.callbacks.EarlyStopping(patience=5,min_delta=0.01,verbose=1)

root_logdir = os.path.join(os.curdir, "housing_logs")
def get_run_logdir(): 
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") 
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

model2.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])

model2.save('reg_housing_1.h5')

model3 = keras.models.Sequential()
model3.add(keras.layers.Dense(200, activation="relu"))
model3.add(keras.layers.Dense(30, activation="relu"))
model3.add(keras.layers.Dense(1))

model3.compile(loss="mean_squared_error", optimizer='sgd')

run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

model3.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir ./housing_logs

model3.save('reg_housing_2.h5,')

model4 = keras.models.Sequential()
model4.add(keras.layers.Dense(50, activation="relu"))
model4.add(keras.layers.Dense(50, activation="relu"))
model4.add(keras.layers.Dense(1))

model4.compile(loss="mean_squared_error", optimizer="sgd")

run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

model4.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb, es])

model4.save("reg_housing_3.h5")

