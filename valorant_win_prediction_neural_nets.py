import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

dataset = pd.read_excel("./dataset.xlsx")

# np.random.shuffle(dataset.values)

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 10, shuffle = False)

print(train_X[:5])

# print(train_X)
# print(train_Y)

# The model
model = keras.Sequential([
        keras.layers.Dense(96, input_shape = (6,), activation = "relu"),
        keras.layers.Dense(48, activation="relu"),
        keras.layers.Dense(2, activation = "softmax")
    ]
)

model.compile(
    optimizer = "adam",
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ["accuracy"]
)

model.fit(train_X, train_Y, batch_size = 10, epochs=20)

# Evaluation
print("EVALUATION")
model.evaluate(test_X, test_Y)