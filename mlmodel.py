import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class EWasteMLModel:
    def __init__(self, db_path: str = "ewaste_flow.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        
    def prepare_data(self):
        """Load and prepare data for machine learning"""
        print("🤖 Preparing data for machine learning...")
        
        # Load data with features
        df = pd.read_sql_query('''
            SELECT d.*, l.name as location_name, w.type_name as waste_type_name
            FROM drop_offs d
            JOIN locations l ON d.location_id = l.location_id
            JOIN waste_types w ON d.waste_type_id = w.waste_type_id
        ''', self.conn)
        
        print(f"   Loaded {len(df)} records")
        
        # Create features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        df['is_event'] = df['event_flag'].astype(int)
        
        # Encode categorical variables
        categorical_cols = ['season', 'weather_conditions', 'location_name', 'waste_type_name']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Select features for training
        feature_cols = [
            'location_id', 'waste_type_id', 'day_of_week', 'is_weekend', 
            'is_event', 'month', 'season_encoded', 'weather_conditions_encoded',
            'location_name_encoded', 'waste_type_name_encoded'
        ]
        
        X = df[feature_cols]
        y = df['drop_quantity']
        
        print(f"   Created {len(feature_cols)} features")
        return X, y, df
    
    def train_model(self, X, y):
        """Train Random Forest model"""
        print("🌳 Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   Training set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"📊 Model Performance:")
        print(f"   Training MAE: {train_mae:.2f}")
        print(f"   Test MAE: {test_mae:.2f}")
        print(f"   Training RMSE: {train_rmse:.2f}")
        print(f"   Test RMSE: {test_rmse:.2f}")
        print(f"   Training R²: {train_r2:.3f}")
        print(f"   Test R²: {test_r2:.3f}")
        
        return X_train, X_test, y_train, y_test, y_pred_test
    
    def feature_importance_analysis(self, X):
        """Analyze feature importance"""
        print(f"🔍 Feature Importance Analysis:")
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("   Top 5 most important features:")
        for i, row in feature_importance.head().iterrows():
            print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")
        
        return feature_importance
    
    def create_predictions_plot(self, y_test, y_pred):
        """Create actual vs predicted plot"""
        plt.figure(figsize=(10, 6))
        
        # Sample data for better visualization
        if len(y_test) > 200:
            indices = np.random.choice(len(y_test), 200, replace=False)
            y_test_sample = y_test.iloc[indices]
            y_pred_sample = y_pred[indices]
        else:
            y_test_sample = y_test
            y_pred_sample = y_pred
        
        plt.scatter(y_test_sample, y_pred_sample, alpha=0.6, color='blue')
        plt.plot([y_test_sample.min(), y_test_sample.max()], 
                [y_test_sample.min(), y_test_sample.max()], 'r--', lw=2)
        plt.xlabel('Actual Drop Quantity')
        plt.ylabel('Predicted Drop Quantity')
        plt.title('Random Forest: Actual vs Predicted E-Waste Drop-offs')
        plt.grid(True, alpha=0.3)
        
        # Add R² score to plot
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('ml_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Predictions plot saved as 'ml_predictions.png'")
    
    def make_sample_predictions(self):
        """Make sample predictions for demonstration"""
        print(f"🔮 Sample Predictions:")
        
        # Create sample scenarios
        scenarios = [
            {'location_id': 1, 'waste_type_id': 1, 'day_of_week': 0, 'season': 'Spring'},  # Monday, Smartphones, Spring
            {'location_id': 2, 'waste_type_id': 3, 'day_of_week': 5, 'season': 'Summer'},  # Saturday, Batteries, Summer
            {'location_id': 3, 'waste_type_id': 2, 'day_of_week': 3, 'season': 'Fall'},   # Thursday, Laptops, Fall
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            # Create feature vector (simplified)
            features = np.array([[
                scenario['location_id'], scenario['waste_type_id'], 
                scenario['day_of_week'], 1 if scenario['day_of_week'] >= 5 else 0,
                0, 6, 0, 0, scenario['location_id']-1, scenario['waste_type_id']-1
            ]])
            
            prediction = self.model.predict(features)[0]
            print(f"   Scenario {i}: {prediction:.1f} units predicted")
    
    def close(self):
        self.conn.close()

# Run machine learning pipeline
if __name__ == "__main__":
    print("🚀 E-WASTEFLOW+ MACHINE LEARNING")
    print("=" * 50)
    
    ml_model = EWasteMLModel()
    
    # Prepare data
    X, y, df = ml_model.prepare_data()
    
    # Train model
    X_train, X_test, y_train, y_test, y_pred = ml_model.train_model(X, y)
    
    # Analyze feature importance
    feature_importance = ml_model.feature_importance_analysis(X)
    
    # Create visualizations
    ml_model.create_predictions_plot(y_test, y_pred)
    
    # Make sample predictions
    ml_model.make_sample_predictions()
    
    ml_model.close()
    print("\n🎉 Machine Learning analysis complete!")