import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to the standardized data
pca = PCA(n_components=2)              # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# Print the explained variance
print("Explained variance by each component:", pca.explained_variance_ratio_)
print("Sum of explained variance (2 components):", np.sum(pca.explained_variance_ratio_))

# Reconstruct the data from the reduced PCA components
X_reconstructed = pca.inverse_transform(X_pca)

# Plot the original and reduced data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.6, label='Original Data')
plt.title('Original Iris Data (First Two Features)')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.6, label='Reconstructed Data (2 Components)', color='r')
plt.title('Reconstructed Iris Data from PCA (2 Components)')
plt.xlabel('Feature 1 (Reconstructed)')
plt.ylabel('Feature 2 (Reconstructed)')
plt.axis('equal')

plt.tight_layout()
plt.show()
