from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import pandas as pd
import plotly.graph_objs as go

app = FastAPI()

# Load the CSVs once and extract unique suburbs
df1 = pd.read_csv('database/csv1.csv')
df2 = pd.read_csv('database/csv2.csv')

# Ensure 'Suburb' column exists and has no missing values
df1 = df1.dropna(subset=['Suburb'])

# Get unique suburbs from the 'Suburb' column
suburb_list = df1['Suburb'].unique()

# HTML Form to collect start date, end date, and dynamically populate suburb (dropdown)
@app.get("/", response_class=HTMLResponse)
async def home():
    # Generate the options for the suburb dropdown dynamically
    suburb_options = ''.join([f'<option value="{suburb}">{suburb}</option>' for suburb in suburb_list])
    
    html_content = f"""
    <html>
        <head>
            <title>Housing Price Plot</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f4f4f4;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                form {{
                    max-width: 500px;
                    margin: auto;
                    padding: 20px;
                    background-color: #fff;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }}
                label {{
                    font-weight: bold;
                    display: block;
                    margin: 10px 0 5px;
                }}
                input[type="date"],
                select {{
                    width: 100%;
                    padding: 8px;
                    margin: 8px 0;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    box-sizing: border-box;
                }}
                input[type="submit"] {{
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 15px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                }}
                input[type="submit"]:hover {{
                    background-color: #45a049;
                }}
            </style>
        </head>
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

# Plot route that accepts query parameters for start date, end date, and suburb
@app.get("/plot")
async def plot(
    start_date: str = Query(..., description="Enter the start date (YYYY-MM-DD)"), 
    end_date: str = Query(..., description="Enter the end date (YYYY-MM-DD)"),
    suburb: str = Query(..., description="Enter the suburb name")
):
    # Filter data by start date, end date, and suburb
    df1['Date'] = pd.to_datetime(df1['Date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df_filtered = df1[(df1['Date'] >= start_date) & (df1['Date'] <= end_date) & (df1['Suburb'].str.lower() == suburb.lower())]

    if df_filtered.empty:
        return HTMLResponse("<h1>No data available for the selected date range and suburb.</h1>")
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta']

    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_filtered['Date'],
        y=df_filtered['Price'],
        marker=dict(color=colors * (len(df_filtered) // len(colors) + 1)),
        name='Price'
    ))

    # Set title and labels
    fig.update_layout(
        title=f"Housing Price Bar Chart for {suburb.capitalize()} ({start_date.date()} to {end_date.date()})",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend"
    )

    # Return the plot in HTML
    return HTMLResponse(fig.to_html(full_html=False))
