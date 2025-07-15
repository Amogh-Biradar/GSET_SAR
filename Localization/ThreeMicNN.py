import torch
import torch.nn as nn
import torch.optim as optim
from ThreeMicSimData import get3micSimData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. Get data
N = 5000  # Sample size
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
    def __init__(self, input_size=2, hidden_sizes=[64, 256, 1024, 256, 64], output_size=2, dropout_rate=0.2):
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


# Enhanced optimizer with advanced parameter groups and techniques
param_groups = [

    {
        'params': model.input_norm.parameters(), 
        'lr': 1e-4,
        'weight_decay': 1e-5,  # Lower weight decay for normalization layers
        'betas': (0.9, 0.999),
        'eps': 1e-8
    },
    {
        'params': model.layers.parameters(), 
        'lr': 2e-3,  # Increased base learning rate
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-8
    },
    {
        'params': model.output_layers.parameters(), 
        'lr': 8e-4,  # Slightly increased output LR
        'weight_decay': 5e-5,  # Lower weight decay for output layers
        'betas': (0.9, 0.999),
        'eps': 1e-8
    }
]

# Primary optimizer with enhanced AdamW
optimizer = optim.AdamW(param_groups, amsgrad=True)  # Enable AMSGrad for better convergence

# Secondary optimizer for fine-tuning (will be used in later epochs)
secondary_optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True, weight_decay=1e-5)

# Advanced learning rate scheduler with warm-up
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs=10, max_epochs=300, eta_min=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * warmup_factor
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.eta_min + (self.base_lrs[i] - self.eta_min) * cosine_factor

# Initialize custom scheduler
custom_scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=15, max_epochs=300, eta_min=1e-7)

# Additional scheduler for fine-tuning phases
fine_tune_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7
)

# 4. Enhanced Training Loop with advanced optimizer techniques
epochs = 300
best_loss = float('inf')
patience = 120  # Increased patience for better convergence
patience_counter = 0
train_losses = []
val_losses = []
learning_rates = []

# Enhanced gradient clipping with adaptive scaling
max_grad_norm = 1.0
grad_norm_history = []

# Optimizer switching parameters
switch_to_sgd_epoch = 200  # Switch to SGD for fine-tuning
current_optimizer = optimizer
use_sgd = False

# Loss smoothing for better early stopping
loss_smoothing_window = 5
val_loss_history = []

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    num_batches = 0
    
    # Switch to SGD optimizer for fine-tuning in later epochs
    if epoch >= switch_to_sgd_epoch and not use_sgd:
        print(f"Switching to SGD optimizer at epoch {epoch+1} for fine-tuning")
        current_optimizer = secondary_optimizer
        use_sgd = True
    
    # Enhanced mini-batch training
    batch_size = 256
    num_samples = len(X_train)
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_X = X_train[i:end_idx]
        batch_y = y_train[i:end_idx]
        
        current_optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Adaptive gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_norm_history.append(grad_norm.item())
        
        # Adjust gradient clipping based on gradient norm history
        if len(grad_norm_history) > 100:
            avg_grad_norm = np.mean(grad_norm_history[-100:])
            if avg_grad_norm > 2.0:
                max_grad_norm = min(max_grad_norm * 0.95, 2.0)  # Decrease clipping threshold
            elif avg_grad_norm < 0.5:
                max_grad_norm = max(max_grad_norm * 1.05, 0.1)  # Increase clipping threshold
        
        current_optimizer.step()
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
        val_loss_history.append(val_loss.item())
    
    # Update learning rate schedulers
    if not use_sgd:
        custom_scheduler.step(epoch)
        fine_tune_scheduler.step(val_loss)
    else:
        # Simple decay for SGD phase
        if epoch % 20 == 0:
            for param_group in current_optimizer.param_groups:
                param_group['lr'] *= 0.95
    
    # Record current learning rate
    current_lr = current_optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    
    # Enhanced early stopping with loss smoothing
    if len(val_loss_history) >= loss_smoothing_window:
        smoothed_val_loss = np.mean(val_loss_history[-loss_smoothing_window:])
        improvement_threshold = 1e-7 if not use_sgd else 1e-8  # Stricter threshold for SGD phase
        
        if smoothed_val_loss < best_loss - improvement_threshold:
            best_loss = smoothed_val_loss
            patience_counter = 0
            # Save best model with enhanced checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': current_optimizer.state_dict(),
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'epoch': epoch,
                'loss': best_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates,
                'use_sgd': use_sgd,
                'grad_norm_history': grad_norm_history[-1000:]  # Keep last 1000 gradient norms
            }, 'tdoa_corrector_best_model.pth')
        else:
            patience_counter += 1
    
    # Dynamic learning rate adjustment based on loss plateau
    if epoch > 50 and not use_sgd:
        recent_losses = val_loss_history[-20:] if len(val_loss_history) >= 20 else val_loss_history
        if len(recent_losses) >= 10:
            loss_variance = np.var(recent_losses)
            if loss_variance < 1e-10:  # Very flat loss curve
                for param_group in current_optimizer.param_groups:
                    param_group['lr'] *= 1.2  # Increase LR to escape plateau
                print(f"Increased learning rate due to loss plateau at epoch {epoch+1}")
    
    if (epoch + 1) % 15 == 0:
        recent_grad_norm = np.mean(grad_norm_history[-10:]) if len(grad_norm_history) >= 10 else 0
        optimizer_name = "SGD" if use_sgd else "AdamW"
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}, "
              f"LR: {current_lr:.2e}, Grad Norm: {recent_grad_norm:.3f}, Optimizer: {optimizer_name}")
    
    # Early stopping
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load the best model
checkpoint = torch.load('tdoa_corrector_best_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"\nBest model loaded with validation loss: {best_loss:.6f}")

# Plot enhanced training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss', alpha=0.7)
plt.plot(val_losses, label='Validation Loss', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 3, 2)
plt.plot(learning_rates, alpha=0.7, color='orange')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 3, 3)
if len(grad_norm_history) > 0:
    # Plot gradient norm history (smoothed)
    smoothed_grad_norms = np.convolve(grad_norm_history, np.ones(10)/10, mode='valid')
    plt.plot(smoothed_grad_norms, alpha=0.7, color='red')
    plt.xlabel('Training Step')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm History (Smoothed)')
    plt.grid(True, alpha=0.3)

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
