from django.shortcuts import render
from .forms import PredictionForm
from .models import AQIData
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .models import AQIPrediction
import time
from datetime import datetime

def prepare_features(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    # Add cyclical encoding for temporal features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    return df

def create_sequences(X, y, sequence_length=24):
    """Create sequences for LSTM input"""
    Xs, ys = [], []
    for i in range(len(X) - sequence_length):
        Xs.append(X[i:(i + sequence_length)])
        ys.append(y[i + sequence_length])
    return np.array(Xs), np.array(ys)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        self.fc = nn.Linear(hidden_size, 2)  # Simplified architecture

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(model_type):
    try:
        data = AQIData.objects.all().values()
        if not data:
            print("No data found in AQIData")
            return None
            
        df = pd.DataFrame(data)
        df = prepare_features(df)

        base_features = ['hour', 'day', 'month', 'day_of_week']
        lstm_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                        'day', 'month', 'day_of_week']
        
        # Scale features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        if model_type == 'lstm':
            # Define sequence length at the start
            sequence_length = 24
            
            # Check CUDA and print more detailed info
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")
            print(f"Using device: {device}")

            # Optimize data preparation
            print("Preparing data...")
            X = df[lstm_features].values
            y = np.stack([df['pm25'].values, df['o3'].values], axis=1)
            
            # Scale features and targets
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y)
            
            X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
            print(f"Sequence shapes - X: {X_seq.shape}, y: {y_seq.shape}")
            
            # Create dataset and dataloader
            dataset = TimeSeriesDataset(X_seq, y_seq)
            train_loader = DataLoader(
                dataset,
                batch_size=64,
                shuffle=True,
                pin_memory=True
            )
            
            # Initialize model and move to GPU
            model = LSTMModel(
                input_size=len(lstm_features),
                hidden_size=128,
                num_layers=2
            ).to(device)
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            num_epochs = 50
            best_loss = float('inf')
            patience = 5
            patience_counter = 0
            
            print("\nStarting training...")
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                model.train()
                total_loss = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                epoch_time = time.time() - epoch_start_time
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f'Epoch [{epoch+1}/{num_epochs}] - '
                      f'Loss: {avg_loss:.4f} - '
                      f'LR: {current_lr:.6f} - '
                      f'Time: {epoch_time:.2f}s')
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        break
            
            model.load_state_dict(torch.load('best_model.pth', weights_only=True))
            return (model, (scaler_X, scaler_y), sequence_length), lstm_features
                
        elif model_type == 'random_forest':
            X = df[base_features].values
            y_pm25 = df['pm25'].values
            y_o3 = df['o3'].values
            
            # Scale features
            X_scaled = scaler_X.fit_transform(X)
            
            # Initialize and train Random Forest models
            pm25_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            o3_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            # Train the models
            pm25_model.fit(X_scaled, y_pm25)
            o3_model.fit(X_scaled, y_o3)
            
            return (pm25_model, o3_model, scaler_X), base_features

    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        return None

def calculate_overall_aqi(pm25, o3):
    aqi_pm25 = pm25 * 1.1
    aqi_o3 = o3 * 1.2
    return max(aqi_pm25, aqi_o3)

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "Air quality is satisfactory, and air pollution poses little or no risk.", "No health implications; enjoy outdoor activities."
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.", "Sensitive individuals should limit prolonged outdoor exertion."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects. The general public is less likely to be affected.", "Consider reducing prolonged outdoor exertion."
    elif aqi <= 200:
        return "Unhealthy", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.", "Limit prolonged outdoor exertion, especially sensitive groups."
    elif aqi <= 300:
        return "Very Unhealthy", "Health alert: The risk of health effects is increased for everyone.", "Avoid all outdoor activities, if possible."
    else:
        return "Hazardous", "Health warning of emergency conditions: everyone is more likely to be affected.", "Stay indoors and avoid physical activities outside."

def predict_aqi(request):
    try:
        form = PredictionForm(request.POST or None)

        if request.method == 'POST' and form.is_valid():
            prediction_datetime = form.cleaned_data['prediction_datetime']
            model_type = form.cleaned_data['model']

            # Prepare the input data for prediction
            pred_df = pd.DataFrame({
                'datetime': [prediction_datetime]
            })
            pred_df = prepare_features(pred_df)

            if model_type == 'lstm':
                result = train_model(model_type)
                if result is None:
                    return render(request, 'aqi_prediction/prediction_form.html', {
                        'form': form,
                        'error': 'Error training model. Please try again.'
                    })
                    
                (model, (scaler_X, scaler_y), sequence_length), features = result
                
                try:
                    # Get recent data
                    recent_data = AQIData.objects.all().order_by('-datetime')[:sequence_length]
                    if len(recent_data) < sequence_length:
                        return render(request, 'aqi_prediction/prediction_form.html', {
                            'form': form,
                            'error': f'Not enough historical data. Need at least {sequence_length} records.'
                        })
                        
                    recent_df = pd.DataFrame(list(recent_data.values()))
                    recent_df = prepare_features(recent_df)
                    
                    # Prepare sequence
                    X = recent_df[features].values
                    X_scaled = scaler_X.transform(X)
                    X_seq = X_scaled.reshape(1, sequence_length, len(features))
                    
                    # Convert to PyTorch tensor and move to device
                    device = next(model.parameters()).device
                    X_seq_tensor = torch.FloatTensor(X_seq).to(device)
                    
                    # Make prediction
                    model.eval()
                    with torch.no_grad():
                        prediction_scaled = model(X_seq_tensor)
                        prediction = scaler_y.inverse_transform(prediction_scaled.cpu().numpy())
                        pm25_pred, o3_pred = prediction[0]
                        
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
                    return render(request, 'aqi_prediction/prediction_form.html', {
                        'form': form,
                        'error': 'Error making prediction. Please try again.'
                    })
            elif model_type == 'random_forest':
                result = train_model(model_type)
                if result is None:
                    return render(request, 'aqi_prediction/prediction_form.html', {
                        'form': form,
                        'error': 'Error training model. Please try again.'
                    })
                
                (pm25_model, o3_model, scaler_X), features = result
                
                # Prepare features for prediction
                X = pred_df[features].values
                X_scaled = scaler_X.transform(X)
                
                # Make predictions
                pm25_pred = pm25_model.predict(X_scaled)[0]
                o3_pred = o3_model.predict(X_scaled)[0]

            # Calculate overall AQI
            overall_aqi = calculate_overall_aqi(pm25_pred, o3_pred)
            aqi_category, health_message, health_tip = get_aqi_category(overall_aqi)

            # Save the prediction
            new_prediction = AQIPrediction(
                prediction_datetime=prediction_datetime,
                pm25_prediction=pm25_pred,
                o3_prediction=o3_pred,
                overall_aqi=overall_aqi,
                aqi_category=aqi_category,
                model_type=model_type
            )
            new_prediction.save()

            return render(request, 'aqi_prediction/prediction_result.html', {
                'prediction_datetime': prediction_datetime,
                'pm25_prediction': round(pm25_pred, 2),
                'o3_prediction': round(o3_pred, 2),
                'overall_aqi': round(overall_aqi, 2),
                'model_type': model_type,
                'aqi_category': aqi_category,
                'health_message': health_message,
                'health_tip': health_tip
            })

        return render(request, 'aqi_prediction/prediction_form.html', {'form': form})
        
    except Exception as e:
        print(f"Error in predict_aqi view: {str(e)}")
        return render(request, 'aqi_prediction/prediction_form.html', {
            'form': form,
            'error': 'An unexpected error occurred. Please try again.'
        })