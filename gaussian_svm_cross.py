from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Define the training data and labels
X = np.array([[-2, 0], [0, 2], [0, -2], [2, 0], [-1, 0], [1, 0], [0, 1], [0, -1]])
y = np.array([1, 1, 1, 1, 0, 0, 0, 0])  

# Create and fit the model
svm = SVC(kernel='rbf') 
svm.fit(X, y)

# Generate grid to cover the feature space
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Plot decision regions
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.5, cmap='Greens')
colors = ['yellow' if label == 0 else 'blue' for label in y]  # yellow for Class 2, blue for Class 1
plt.scatter(X[:, 0], X[:, 1], c=colors, s=50, edgecolors='k')
plt.title('Decision Region Boundary for Gaussian SVM')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
