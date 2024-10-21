import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Load your Excel data
df = pd.read_excel('data.xlsx')

# Function to convert 'YYYY.MM' format to proper datetime with error handling
def convert_to_date(date_float):
    try:
        year = int(date_float)
        month = int((date_float - year) * 100)
        if 1 <= month <= 12:
            return datetime(year, month, 1)  # Assuming the 1st day of each month
        else:
            raise ValueError(f"Invalid month {month} for date {date_float}")
    except ValueError as ve:
        print(f"Error parsing date {date_float}: {ve}")
        return None  # Return None for invalid dates

# Apply the function to the 'Date' column
df['Date'] = df['Date'].apply(convert_to_date)

# Drop rows where Date is None (invalid dates)
df = df.dropna(subset=['Date'])

# Sort by date to ensure data is in correct order
df = df.sort_values(by='Date')

# Get the latest date and calculate the start date for "Last 1 Year"
end_date = df['Date'].max()
start_date = end_date - relativedelta(years=1)

# Filter the data for the last 1 year
df_last_year = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Starting with $1, calculate the cumulative value for the 'Composite Price' column
def calculate_cumulative_value(data, column_name):
    cumulative = [1]  # Start with $1
    for i in range(1, len(data)):
        percentage_change = (data[column_name].iloc[i] / data[column_name].iloc[i-1]) - 1
        cumulative.append(cumulative[-1] * (1 + percentage_change))
    return cumulative

# Calculate cumulative value for 'Composite Price' for the last 1 year
df_last_year['Cumulative Composite Price'] = calculate_cumulative_value(df_last_year, 'Composite Price')

# Plot the cumulative value over the last 1 year
plt.plot(df_last_year['Date'], df_last_year['Cumulative Composite Price'], color='purple', label='Cumulative Composite Price')

# Plot formatting
plt.xlabel('Date')
plt.ylabel('Cumulative Value of $1')
plt.title('Cumulative Composite Price Over Last 1 Year (Starting with $1)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()