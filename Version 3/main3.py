import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import plotly.graph_objs as go

# Constants
N_STEPS = 60
LOOKUP_STEP = 15
SCALE = True
FEATURE_COLUMNS = ["Price", "Rooms", "Propertycount", "Distance"]  # Relevant columns
BATCH_SIZE = 64
EPOCHS = 5
MODEL_PATH = "saved_model/housing_price_model.keras"  # Path to save/load the model
dataframe_path = 'database/csv1.csv'

app = FastAPI()

# Load the CSV and extract unique suburbs
df1 = pd.read_csv(dataframe_path)
df1['Date'] = pd.to_datetime(df1['Date'])
df1 = df1.dropna(subset=['Suburb', 'Price'])

unique_suburbs = df1['Suburb'].unique()

# Data loading and processing function
def load_and_process_housing_data(file_path, handle_nan='fill', split_ratio=0.8, scale=True):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])

    if handle_nan == 'drop':
        df.dropna(inplace=True)
    elif handle_nan == 'fill':
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    # Select only numerical columns for scaling
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    column_scaler = {}

    if scale:
        scaler = MinMaxScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        column_scaler = {col: scaler for col in numerical_columns}

    # Split into training and testing sets by date
    train_df = df[:int(len(df) * split_ratio)]
    test_df = df[int(len(df) * split_ratio):]

    return {
        'train': train_df,
        'test': test_df,
        'column_scaler': column_scaler,
    }, column_scaler

# Model creation function
def create_model(n_steps, n_features, loss='huber', units=256, cell=GRU, n_layers=2, dropout=0.4, optimizer='adam', bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(cell(units, return_sequences=(n_layers > 1), input_shape=(n_steps, n_features)))
        else:
            model.add(cell(units, return_sequences=(i < n_layers - 1)))
        model.add(Dropout(dropout))

    model.add(Dense(1, activation='linear'))
    model.compile(loss=loss, optimizer=optimizer)
    return model

# Create housing sequences
def create_housing_sequences(data, feature_columns, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[feature_columns].iloc[i:i + n_steps].values)
        y.append(data['Price'].iloc[i + n_steps])  # Assuming 'Price' is the target column
    return np.array(X), np.array(y)

# Multistep prediction function
def multistep_predict_housing(model, data, n_steps, k):
    last_sequence = data['test'][FEATURE_COLUMNS].values[-n_steps:]
    last_sequence = last_sequence.reshape((1, n_steps, len(FEATURE_COLUMNS)))
    predictions = []

    for step in range(k):
        # Make the prediction
        prediction = model.predict(last_sequence)

        # We are only predicting the 'Price' column, so inverse transform only that column
        if SCALE:
            price_scaler = data['column_scaler']['Price']  # Access only the 'Price' scaler
            prediction = price_scaler.inverse_transform(prediction.reshape(-1, 1))

        # Add the predicted price to the list of predictions
        predictions.append(prediction[0][0])

        # Update the last_sequence with the predicted price for the next step
        new_sequence = np.copy(last_sequence)
        new_sequence[:, -1, 0] = prediction[0, 0]  # Update only the 'Price' feature in the sequence
        last_sequence = np.append(new_sequence[:, 1:, :], new_sequence[:, -1:, :], axis=1)

    return predictions

# Function to load or create model
def create_or_load_model(X_train, y_train, X_test, y_test):
    # Check if the model exists
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded from disk.")
    else:
        # Create and train the model
        model = create_model(
            n_steps=N_STEPS,
            n_features=len(FEATURE_COLUMNS),
            loss='huber',
            units=256,
            cell=GRU,
            n_layers=2,
            dropout=0.4,
            optimizer='adam',
            bidirectional=False
        )
        
        # Train the model
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

        # Save the model to disk
        model.save(MODEL_PATH)
        print(f"Model trained and saved to {MODEL_PATH}")

    return model

# HTML form for FastAPI
@app.get("/", response_class=HTMLResponse)
async def home():
    suburb_options = ''.join([f'<option value="{suburb}">{suburb}</option>' for suburb in unique_suburbs])
    html_content = f"""
    <html>
        <head><title>Housing Price Prediction</title></head>
        <body>
            <h1>Enter Start Date, End Date, and Suburb</h1>
            <form action="/plot" method="get">
                <label for="start_date">Start Date:</label>
                <input type="date" id="start_date" name="start_date" required><br><br>
                <label for="end_date">End Date:</label>
                <input type="date" id="end_date" name="end_date" required><br><br>
                <label for="suburb">Suburb:</label>
                <select id="suburb" name="suburb" required>
                    <option value="" disabled selected>Select a suburb</option>
                    {suburb_options}
                </select><br><br>
                <input type="submit" value="Generate Plot">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Plotting route
@app.get("/plot")
async def plot(
    start_date: str = Query(..., description="Enter the start date (YYYY-MM-DD)"), 
    end_date: str = Query(..., description="Enter the end date (YYYY-MM-DD)"),
    suburb: str = Query(..., description="Enter the suburb name")
):
    # Load and process data
    data, column_scalers = load_and_process_housing_data(dataframe_path)
    
    # Create training and test datasets
    X_train, y_train = create_housing_sequences(data['train'], FEATURE_COLUMNS, N_STEPS)
    X_test, y_test = create_housing_sequences(data['test'], FEATURE_COLUMNS, N_STEPS)

    # Load or create the model
    model = create_or_load_model(X_train, y_train, X_test, y_test)

    # Predict future housing prices
    future_steps = LOOKUP_STEP
    predicted_prices = multistep_predict_housing(model, data, N_STEPS, future_steps)

    # Plot actual and predicted prices
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data['test']['Date'],
        y=data['test']['Price'],
        marker=dict(color='blue'),
        name='Actual Price'
    ))

    # Plot predicted prices
    future_dates = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=future_steps)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predicted_prices,
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='red', dash='dash')
    ))

    # Set titles and labels
    fig.update_layout(
        title=f"Housing Price Prediction for {suburb.capitalize()}",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend"
    )

    return HTMLResponse(fig.to_html(full_html=False))
