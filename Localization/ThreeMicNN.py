import torch
import torch.nn as nn
import torch.optim as optim
from ThreeMicSimData import get3micSimData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. Get data
N = 10000  # Sample size
measured, true = get3micSimData(N)

# Convert to NumPy arrays for preprocessing
X = np.array(measured, dtype=np.float32)
y = np.array(true, dtype=np.float32)

# 2. Preprocessing
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# 3. Neural Network with Advanced Architecture
class TDOACorrector(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[256, 128, 64, 32], output_size=2, dropout_rate=0.3):
        super().__init__()
        
        # Input normalization layer
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Build the network layers with residual connections
        self.layers = nn.ModuleList()
        self.residual_connections = []  # Store residual flags separately
        prev_size = input_size
        
        # Hidden layers with batch normalization, dropout, and residual connections
        for i, hidden_size in enumerate(hidden_sizes):
            layer_block = nn.ModuleDict({
                'linear': nn.Linear(prev_size, hidden_size),
                'batch_norm': nn.BatchNorm1d(hidden_size),
                'activation': nn.LeakyReLU(0.2),  # LeakyReLU instead of ReLU
                'dropout': nn.Dropout(dropout_rate)
            })
            
            # Store residual connection flag separately
            has_residual = (prev_size == hidden_size)
            self.residual_connections.append(has_residual)
                
            self.layers.append(layer_block)
            prev_size = hidden_size
        
        # Output layer with smaller final layer
        self.output_layers = nn.Sequential(
            nn.Linear(prev_size, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate * 0.5),  # Reduced dropout for output
            nn.Linear(16, output_size)
        )
        
        # Initialize weights using He initialization (better for LeakyReLU)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)
        
        # Forward through hidden layers with residual connections
        for i, layer_block in enumerate(self.layers):
            identity = x
            
            # Apply transformations
            x = layer_block['linear'](x)
            x = layer_block['batch_norm'](x)
            x = layer_block['activation'](x)
            x = layer_block['dropout'](x)
            
            # Add residual connection if dimensions match
            if self.residual_connections[i] and x.shape == identity.shape:
                x = x + identity
        
        # Output layer
        x = self.output_layers(x)
        return x

model = TDOACorrector()

# Advanced loss function with L1 + L2 combination (Huber-like loss)
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.mae(pred, target)

criterion = CombinedLoss(alpha=0.8)

# Advanced optimizer with different learning rates for different layers
param_groups = [
    {'params': model.input_norm.parameters(), 'lr': 1e-4},  # Lower LR for input norm
    {'params': model.layers.parameters(), 'lr': 1e-3},      # Standard LR for hidden layers
    {'params': model.output_layers.parameters(), 'lr': 5e-4} # Lower LR for output layers
]
optimizer = optim.AdamW(param_groups, weight_decay=1e-4)  # AdamW instead of Adam

# More sophisticated learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

# 4. Advanced Training Loop with multiple improvements
epochs = 300
best_loss = float('inf')
patience = 30
patience_counter = 0
train_losses = []
val_losses = []

# Gradient clipping for stability
max_grad_norm = 1.0

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    num_batches = 0
    
    # Mini-batch training for better gradient estimates
    batch_size = 256
    num_samples = len(X_train)
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_X = X_train[i:end_idx]
        batch_y = y_train[i:end_idx]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        epoch_train_loss += loss.item()
        num_batches += 1
    
    avg_train_loss = epoch_train_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # Validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)
        val_losses.append(val_loss.item())
    
    # Update learning rate scheduler
    scheduler.step()
    
    # Early stopping check with improvement threshold
    improvement_threshold = 1e-6
    if val_loss < best_loss - improvement_threshold:
        best_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'epoch': epoch,
            'loss': best_loss
        }, 'tdoa_corrector_best_model.pth')
    else:
        patience_counter += 1
    
    if (epoch + 1) % 20 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}, LR: {current_lr:.2e}")
    
    # Early stopping
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load the best model
checkpoint = torch.load('tdoa_corrector_best_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"\nBest model loaded with validation loss: {best_loss:.6f}")

# Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', alpha=0.7)
plt.plot(val_losses, label='Validation Loss', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot([optimizer.param_groups[0]['lr'] for _ in range(len(train_losses))], alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()

# Save the trained model
torch.save(model.state_dict(), 'tdoa_corrector_model.pth')
print(f"\nModel saved as 'tdoa_corrector_model.pth'")

# 5. Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions_rescaled = scaler_y.inverse_transform(predictions.numpy())
    y_test_rescaled = scaler_y.inverse_transform(y_test.numpy())

# Optional: print a few predictions
print("\nSample predictions:")
for i in range(5):
    print(f"Measured: {scaler_X.inverse_transform(X_test[i].numpy().reshape(1, -1))[0]}")
    print(f"Predicted True: {predictions_rescaled[i]}")
    print(f"Actual True:    {y_test_rescaled[i]}")
    print("---")

# Create scatter plot
plt.figure(figsize=(12, 6))

# Flatten the data for plotting
X_test_rescaled = scaler_X.inverse_transform(X_test.numpy())
measured_flat = X_test_rescaled.flatten()
true_flat = y_test_rescaled.flatten()
predicted_flat = predictions_rescaled.flatten()

# Plot 1: Original data (measured vs true)
plt.subplot(1, 2, 1)
plt.scatter(true_flat, measured_flat, alpha=0.6, s=30, color='blue', label='Original Data')
min_val = min(min(measured_flat), min(true_flat))
max_val = max(max(measured_flat), max(true_flat))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Match (y=x)')
plt.xlabel('True TDOA (seconds)', fontsize=12)
plt.ylabel('Measured TDOA (seconds)', fontsize=12)
plt.title('Original Data: Measured vs True TDOA', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Plot 2: Model predictions (predicted vs true)
plt.subplot(1, 2, 2)
plt.scatter(true_flat, predicted_flat, alpha=0.6, s=30, color='green', label='Model Predictions')
plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Match (y=x)')
plt.xlabel('True TDOA (seconds)', fontsize=12)
plt.ylabel('Predicted TDOA (seconds)', fontsize=12)
plt.title('Model Predictions: Predicted vs True TDOA', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.show()

# Calculate and print performance metrics
mse_original = np.mean((measured_flat - true_flat) ** 2)
mse_model = np.mean((predicted_flat - true_flat) ** 2)
mae_original = np.mean(np.abs(measured_flat - true_flat))
mae_model = np.mean(np.abs(predicted_flat - true_flat))

print(f"\nPerformance Metrics:")
print(f"Original Data - MSE: {mse_original:.6f}, MAE: {mae_original:.6f}")
print(f"Model Predictions - MSE: {mse_model:.6f}, MAE: {mae_model:.6f}")
print(f"MSE Improvement: {((mse_original - mse_model) / mse_original * 100):.2f}%")
print(f"MAE Improvement: {((mae_original - mae_model) / mae_original * 100):.2f}%")
