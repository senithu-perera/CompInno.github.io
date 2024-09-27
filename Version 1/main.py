from fastapi import FastAPI
import pandas as pd
import plotly.graph_objs as go
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/plot")
async def plot():
    # Load CSVs (replace with actual file paths)
    df1 = pd.read_csv('database/csv1.csv')
    df2 = pd.read_csv('database/csv2.csv')

    # Ensure 'Date' column is in datetime format and sort values
    df1['Date'] = pd.to_datetime(df1['Date'])
    df1 = df1.sort_values('Date')

    # Drop missing values in 'Price' if necessary
    df1 = df1.dropna(subset=['Price'])

    # Create unique colors for each bar (one color per price entry)
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta']

    # If there are multiple price columns, choose the one you want to plot, e.g., 'Price'
    # If you have multiple values per day, consider grouping or averaging them.
    
    # Create the Plotly bar chart
    fig = go.Figure()

    # Bar chart with different colors for each price
    fig.add_trace(go.Bar(
        x=df1['Date'],
        y=df1['Price'],
        marker=dict(color=colors * (len(df1) // len(colors) + 1)),  # Cycle through colors
        name='Price',
    ))

    # Set title and labels
    fig.update_layout(
        title="Housing Price Bar Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        font=dict(family="Arial, monospace", size=18, color="RebeccaPurple"),
        xaxis_tickformat='%Y-%m-%d',  # Format the date for better readability
    )

    # Return HTML containing the plot
    return HTMLResponse(fig.to_html(full_html=False))
