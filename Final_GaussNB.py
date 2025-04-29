import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.datasets import fashion_mnist
import pickle

# 1. Load Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Flatten the 28x28 images into 784-dimensional vectors
X_train_flat = X_train.reshape(-1, 28*28)
X_test_flat = X_test.reshape(-1, 28*28)

# 2. Visualize a few samples from the Fashion-MNIST dataset
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()

# 3. Define and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train_flat, y_train)

# 4. Generate predictions on the test set
y_pred = model.predict(X_test_flat)
print("Predicted labels for the first 50 test samples:", y_pred[:50])

# 5. Visualize some predictions on test data
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f"True: {y_test[i]} | Pred: {y_pred[i]}")
    plt.axis('off')
plt.show()

# 6. Calculate the prediction probabilities
y_prob = model.predict_proba(X_test_flat)
print("Prediction probabilities for the first 5 test samples:\n", y_prob[:5].round(2))

# 7. Save the model using pickle
filename = 'fashion_mnist_NB_model.sav'
pickle.dump(model, open(filename, 'wb'))

# 8. Load the model from disk and use it for predictions
loaded_model = pickle.load(open(filename, 'rb'))
y_pred_loaded = loaded_model.predict(X_test_flat)

# 9. Evaluate the loaded model
accuracy = loaded_model.score(X_test_flat, y_test)
print(f"Accuracy of the loaded model: {accuracy * 100:.2f}%")

