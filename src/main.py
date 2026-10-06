import torch
import torch.nn as nn
import torch.optim as optim

"""
optimizer.zero_grad():
    Clears existing gradients so they don't accumulate across
    training iterations.

loss.backward():
    Triggers PyTorch's autograd engine to calculate
        $frac{partial text{loss}}{partial w}$
    for every trainable parameter.

optimizer.step():
    Adjusts model parameters using calculated gradients and
    the learning rate.
"""


def main():
    """
    This function demonstrates a simple linear regression model using PyTorch.
    """
    # 1. Create synthetic data: y = 2x + 1
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    y = torch.tensor([[3.0], [5.0], [7.0], [9.0]], dtype=torch.float32)

    # 2. Define a simple Linear Model (1 input feature -> 1 output feature)
    model = nn.Linear(in_features=1, out_features=1)

    # 3. Define Loss Function and Optimizer
    criterion = nn.MSELoss()  # Mean Squared Error
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 4. Training Loop (100 epochs)
    for epoch in range(100):
        # Forward pass
        predictions = model(X)

        # Compute loss
        loss = criterion(predictions, y)

        # Backward pass: zero gradients, backprop, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")

    # 5. Test prediction for x = 5.0 (expected y ≈ 11.0)
    new_x = torch.tensor([[5.0]], dtype=torch.float32)
    predicted_y = model(new_x)
    print(f"\nPrediction for x = 5.0: {predicted_y.item():.4f}")


if __name__ == "__main__":
    main()
