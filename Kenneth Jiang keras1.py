import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

y = to_categorical(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()

# Add input and hidden layers
model.add(Dense(units=10, activation='relu', input_dim=X.shape[1]))
model.add(Dense(units=8, activation='relu'))

# Add output layer with 3 units (for 3 classes) and softmax activation
model.add(Dense(units=3, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, epochs=100, verbose=0)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy}")

predictions = model.predict(X_test)
print("\nPredictions (softmax probabilities):\n", predictions)

predicted_classes = np.argmax(predictions, axis=1)
print("\nPredicted class labels:\n", predicted_classes)
