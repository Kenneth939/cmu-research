import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from tensorflow.keras.datasets import fashion_mnist

# Load the Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocess the data
# Flatten the 28x28 images into vectors of size 784
train_images_flat = train_images.reshape((train_images.shape[0], 28 * 28))
test_images_flat = test_images.reshape((test_images.shape[0], 28 * 28))

# Use a subset of the data for faster computation
n_samples = 1000
train_images_subset = train_images_flat[:n_samples]
train_labels_subset = train_labels[:n_samples]

# For linear regression, let's use a single pixel feature (e.g., pixel 350)
x = train_images_subset[:, 350]  # Selecting pixel 350 as the feature
y = train_labels_subset  # Use labels as the target (class labels)

# Scatter plot of pixel 350 values vs. labels
plt.scatter(x, y)
plt.title("Pixel 350 values vs. Fashion-MNIST Labels")
plt.show()

# Define and fit the linear regression model
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)

# Generate predictions for the data
xfit = np.linspace(0, 255, 1000)  # Pixel values range from 0 to 255 (grayscale)
yfit = model.predict(xfit[:, np.newaxis])

# Plot original data and model predictions
plt.scatter(x, y)
plt.plot(xfit, yfit, 'r.', label='Model Prediction')
plt.title("Linear Regression on Fashion-MNIST")
plt.legend()
plt.show()

# Print model results (slope and intercept)
print("Model slope: ", model.coef_[0])
print("Model intercept:", model.intercept_)

