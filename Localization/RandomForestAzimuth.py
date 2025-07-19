import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib
from audioSim import Mic, Environment, Wave, getEstTDOA, getTrueTDOA

class AzimuthRandomForest:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Random Forest classifier for azimuth prediction from TDOA values
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
        """
        self.azimuth_bins = np.array([-150., -120., -90., -60., -30., 0., 30., 60., 90., 120., 150., 180.])
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def _digitize_azimuth(self, true_azimuth_deg):
        """
        Convert continuous azimuth to discrete bin index
        
        Args:
            true_azimuth_deg: True azimuth angle in degrees
            
        Returns:
            Bin index (0-11)
        """
        # Wrap azimuth to [-180, 180] range
        wrapped_azimuth = ((true_azimuth_deg + 180) % 360) - 180
        
        # Find the closest bin
        bin_idx = np.digitize(wrapped_azimuth, self.azimuth_bins) - 1
        
        # Handle edge cases
        bin_idx = np.clip(bin_idx, 0, len(self.azimuth_bins) - 1)
        
        return bin_idx
    
    def generate_training_data(self, n_samples=10000, max_radius=100):
        """
        Generate synthetic training data using your simulation functions
        
        Args:
            n_samples: Number of training samples to generate
            max_radius: Maximum radius for sound source placement
            
        Returns:
            X: TDOA features [n_samples, 2]
            y: Azimuth bin labels [n_samples]
            true_azimuths: Continuous azimuth values for analysis
        """
        print(f"Generating {n_samples} training samples...")
        
        # Set up microphones (same as your configuration)
        mics = [Mic((0, 0), 10000), Mic((0.05, 0), 10000), Mic((0.025, 0.0433), 10000)]
        mic_positions = [[0, 0], [0.05, 0], [0.025, 0.0433]]
        
        X = []  # TDOA features
        y = []  # Azimuth bin labels
        true_azimuths = []  # For analysis
        
        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"Generated {i}/{n_samples} samples...")
            
            # Generate random source position
            r = max_radius * np.sqrt(np.random.uniform(0, 1))
            theta_true = np.random.uniform(-180, 180)  # True azimuth in degrees
            
            # Convert to Cartesian coordinates
            x_pos = r * np.cos(np.radians(theta_true))
            y_pos = r * np.sin(np.radians(theta_true))
            
            # Create environment
            env = Environment(mics, Wave((x_pos, y_pos)))
            
            # Get TDOA estimates (using the upper bound for consistency)
            tdoa1 = getEstTDOA(mics[0], mics[1], env.getWave())[1]  # mic1 - mic0
            tdoa2 = getEstTDOA(mics[0], mics[2], env.getWave())[1]  # mic2 - mic0
            
            # Scale TDOAs to microseconds for better numerical stability
            tdoa1_us = tdoa1 * 1e6
            tdoa2_us = tdoa2 * 1e6
            
            # Get true azimuth bin
            azimuth_bin = self._digitize_azimuth(theta_true)
            
            X.append([tdoa1_us, tdoa2_us])
            y.append(azimuth_bin)
            true_azimuths.append(theta_true)
        
        X = np.array(X)
        y = np.array(y)
        true_azimuths = np.array(true_azimuths)
        
        print(f"Training data generated: {X.shape[0]} samples")
        print(f"TDOA1 range: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}] μs")
        print(f"TDOA2 range: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}] μs")
        print(f"Azimuth bin distribution: {np.bincount(y)}")
        
        return X, y, true_azimuths
    
    def train(self, X, y):
        """
        Train the Random Forest model
        
        Args:
            X: TDOA features [n_samples, 2]
            y: Azimuth bin labels [n_samples]
        """
        print("Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.rf_model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.rf_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training completed!")
        print(f"Test Accuracy: {accuracy:.3f}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Detailed classification report
        print("\nClassification Report:")
        target_names = [f"Bin_{i}: {self.azimuth_bins[i]:.0f}°" for i in range(len(self.azimuth_bins))]
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Feature importance
        importance = self.rf_model.feature_importances_
        print(f"\nFeature Importance:")
        print(f"TDOA1 (mic1-mic0): {importance[0]:.3f}")
        print(f"TDOA2 (mic2-mic0): {importance[1]:.3f}")
        
        return X_test_scaled, y_test, y_pred
    
    def predict(self, tdoa1_us, tdoa2_us):
        """
        Predict azimuth bin from TDOA values
        
        Args:
            tdoa1_us: TDOA between mic1 and mic0 in microseconds
            tdoa2_us: TDOA between mic2 and mic0 in microseconds
            
        Returns:
            predicted_bin: Bin index (0-11)
            predicted_azimuth: Azimuth angle in degrees
            confidence: Prediction probability
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare input
        X_input = np.array([[tdoa1_us, tdoa2_us]])
        X_scaled = self.scaler.transform(X_input)
        
        # Predict
        predicted_bin = self.rf_model.predict(X_scaled)[0]
        probabilities = self.rf_model.predict_proba(X_scaled)[0]
        confidence = probabilities[predicted_bin]
        
        # Convert bin to azimuth angle
        predicted_azimuth = self.azimuth_bins[predicted_bin]
        
        return predicted_bin, predicted_azimuth, confidence
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f"{int(az)}°" for az in self.azimuth_bins],
                   yticklabels=[f"{int(az)}°" for az in self.azimuth_bins])
        plt.title('Azimuth Classification Confusion Matrix')
        plt.xlabel('Predicted Azimuth Bin')
        plt.ylabel('True Azimuth Bin')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_space(self, X, y):
        """Plot TDOA feature space colored by azimuth bins"""
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab20', alpha=0.6, s=20)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Azimuth Bin')
        cbar.set_ticks(range(len(self.azimuth_bins)))
        cbar.set_ticklabels([f"{int(az)}°" for az in self.azimuth_bins])
        
        plt.xlabel('TDOA1 (mic1-mic0) [μs]')
        plt.ylabel('TDOA2 (mic2-mic0) [μs]')
        plt.title('TDOA Feature Space by Azimuth Bin')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'rf_model': self.rf_model,
            'scaler': self.scaler,
            'azimuth_bins': self.azimuth_bins
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.rf_model = model_data['rf_model']
        self.scaler = model_data['scaler']
        self.azimuth_bins = model_data['azimuth_bins']
        self.is_trained = True
        print(f"Model loaded from {filepath}")

def main():
    """Main training and evaluation pipeline"""
    # Initialize model
    rf_azimuth = AzimuthRandomForest(n_estimators=200, max_depth=20, random_state=42)
    
    # Generate training data
    X, y, true_azimuths = rf_azimuth.generate_training_data(n_samples=20000, max_radius=100)
    
    # Train model
    X_test, y_test, y_pred = rf_azimuth.train(X, y)
    
    # Visualizations
    print("\nGenerating visualizations...")
    rf_azimuth.plot_confusion_matrix(y_test, y_pred)
    rf_azimuth.plot_feature_space(X[:5000], y[:5000])  # Plot subset for clarity
    
    # Save model
    rf_azimuth.save_model('azimuth_rf_model.pkl')
    
    # Test predictions
    print("\nTesting predictions...")
    test_cases = [
        (50.0, 30.0),   # Example TDOA values in microseconds
        (-20.0, 40.0),
        (10.0, -15.0),
        (0.0, 0.0)
    ]
    
    for i, (tdoa1, tdoa2) in enumerate(test_cases):
        bin_idx, azimuth, confidence = rf_azimuth.predict(tdoa1, tdoa2)
        print(f"Test {i+1}: TDOA=({tdoa1:.1f}, {tdoa2:.1f}) μs → "
              f"Bin {bin_idx}, Azimuth {azimuth:.0f}°, Confidence {confidence:.3f}")
    
    return rf_azimuth

if __name__ == "__main__":
    # Run the training pipeline
    model = main()