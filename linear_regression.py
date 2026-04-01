import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
data = pd.read_csv("Housing.csv")
print(data.head())
print("File loaded successfully")

# Example: using 'area' to predict 'price'
X = data["area"].values
Y = data["price"].values

# -----------------------------
# STEP 2: Normalize Data
# -----------------------------
X = X / np.max(X)
Y = Y / np.max(Y)

# -----------------------------
# STEP 3: Initialize Parameters
# -----------------------------
w = 0
b = 0
learning_rate = 0.1
epochs = 100

n = len(X)
losses = []

# -----------------------------
# STEP 4: Gradient Descent
# -----------------------------
for i in range(epochs):

    # Prediction
    Y_pred = w * X + b

    # Loss (MSE)
    loss = (1/n) * np.sum((Y - Y_pred)**2)
    losses.append(loss)

    # Gradients
    dw = (-2/n) * np.sum(X * (Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)

    # Update
    w = w - learning_rate * dw
    b = b - learning_rate * db

# -----------------------------
# STEP 5: Convert to Real Scale
# -----------------------------
w_real = w * (np.max(Y) / np.max(X))
b_real = b * np.max(Y)

# -----------------------------
# STEP 6: Results
# -----------------------------
print("Final weight (w):", w_real)
print("Final bias (b):", b_real)

print("\nFinal Equation:")
print(f"Price = {w_real:.2f} * Area + {b_real:.2f}")

# -----------------------------
# STEP 7: Plot Loss
# -----------------------------
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.show()