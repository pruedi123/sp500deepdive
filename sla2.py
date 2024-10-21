import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Title for the Streamlit app
st.title("Standard and Poor's 500 Index Data")

# Load your local Excel file directly
df = pd.read_excel('data.xlsx')

# Ensure 'Date Fraction' column is of type float
df['Date Fraction'] = df['Date Fraction'].astype(float)

# Function to convert 'Date Fraction' to datetime
def date_fraction_to_datetime(date_fraction):
    year = int(date_fraction)
    fractional_part = date_fraction - year
    month = int(round(fractional_part * 12 + 0.5))
    # Handle edge cases where month might be 0 or 13
    if month < 1:
        month = 1
    elif month > 12:
        month = 12
    date_obj = datetime(year, month, 1)
    return date_obj

# Apply the function to create a new 'Date' column
df['Date'] = df['Date Fraction'].apply(date_fraction_to_datetime)

# Define the last valid date
last_good_date = datetime(2024, 9, 1)

# Exclude dates beyond September 2024
df = df[df['Date'] <= last_good_date]

# Extract unique years from the 'Date' column
years = sorted(df['Date'].dt.year.unique())

# Month mapping: Display full names but convert to numeric month strings
month_mapping = {
    'January': '01',
    'February': '02',
    'March': '03',
    'April': '04',
    'May': '05',
    'June': '06',
    'July': '07',
    'August': '08',
    'September': '09',
    'October': '10',
    'November': '11',
    'December': '12'
}

# Sidebar for date selection
st.sidebar.header("Select Dates")

# Add a dropdown with predefined date ranges
date_options = [
    'Custom Period', 'Last 1 Year', 'Last 3 Years', 'Last 5 Years', 'Last 10 Years',
    'Last 15 Years', 'Last 20 Years', 'Last 25 Years', 'Last 30 Years',
    'Last 35 Years', 'Last 40 Years', 'Since WWII'
]
selected_range = st.sidebar.selectbox('Select Time Period', date_options, index=0)  # Default to 'Custom Period'

# Set the latest date as September 2024
end_date_default = datetime(2024, 9, 1)
end_year_default = 2024
end_month_default = '09'  # September

# Month names list for dropdowns
month_names = list(month_mapping.keys())

# Initialize variables for begin and end dates
begin_year = None
begin_month_name = None
end_year = None
end_month_name = None

# Widgets for selecting begin and end dates (only for custom period)
if selected_range == 'Custom Period':
    begin_year = st.sidebar.selectbox('Select the beginning year', options=years, index=years.index(1984))
    begin_month_name = st.sidebar.selectbox('Select the beginning month', options=month_names, index=9)  # October is at index 9

    end_year = st.sidebar.selectbox('Select the ending year', options=years, index=years.index(end_year_default))
    end_month_name = st.sidebar.selectbox('Select the ending month', options=month_names, index=8)  # September is at index 8

# Add the "Run Selection" button
run_button = st.sidebar.button('Run Selection')

# Execute the code only when the button is clicked
if run_button:
    if selected_range != 'Custom Period':
        # Fixed end date
        end_date_obj = end_date_default

        # Determine the begin date based on the selection
        if selected_range == 'Last 1 Year':
            begin_date_obj = end_date_obj - relativedelta(years=1) + relativedelta(months=1)
        elif selected_range == 'Last 3 Years':
            begin_date_obj = end_date_obj - relativedelta(years=3) + relativedelta(months=1)
        elif selected_range == 'Last 5 Years':
            begin_date_obj = end_date_obj - relativedelta(years=5) + relativedelta(months=1)
        elif selected_range == 'Last 10 Years':
            begin_date_obj = end_date_obj - relativedelta(years=10) + relativedelta(months=1)
        elif selected_range == 'Last 15 Years':
            begin_date_obj = end_date_obj - relativedelta(years=15) + relativedelta(months=1)
        elif selected_range == 'Last 20 Years':
            begin_date_obj = end_date_obj - relativedelta(years=20) + relativedelta(months=1)
        elif selected_range == 'Last 25 Years':
            begin_date_obj = end_date_obj - relativedelta(years=25) + relativedelta(months=1)
        elif selected_range == 'Last 30 Years':
            begin_date_obj = end_date_obj - relativedelta(years=30) + relativedelta(months=1)
        elif selected_range == 'Last 35 Years':
            begin_date_obj = end_date_obj - relativedelta(years=35) + relativedelta(months=1)
        elif selected_range == 'Last 40 Years':
            begin_date_obj = end_date_obj - relativedelta(years=40) + relativedelta(months=1)
        elif selected_range == 'Since WWII':
            begin_date_obj = datetime(1945, 10, 1)  # Start from October 1945

        # Extract begin_year and begin_month from begin_date_obj
        begin_year = begin_date_obj.year
        begin_month = f"{begin_date_obj.month:02d}"  # Ensure two-digit month
        begin_month_name = [name for name, num in month_mapping.items() if num == begin_month][0]

        # Set end_year and end_month_name
        end_year = end_date_obj.year
        end_month = f"{end_date_obj.month:02d}"
        end_month_name = [name for name, num in month_mapping.items() if num == end_month][0]

        # Display the calculated dates in the sidebar
        st.sidebar.write(f"**Begin Date:** {begin_month_name} {begin_year}")
        st.sidebar.write(f"**End Date:** {end_month_name} {end_year}")

    else:
        # Custom Period: Use the values from the widgets
        # Convert month names to the numeric format
        begin_month = month_mapping[begin_month_name]
        end_month = month_mapping[end_month_name]

        # Ensure that end date does not go beyond September 2024
        if int(end_year) > end_year_default or (int(end_year) == end_year_default and int(end_month) > int(end_month_default)):
            st.warning(f"The last usable date is September 2024. Adjusting end date to September 2024.")
            end_year = end_year_default
            end_month = end_month_default
            end_month_name = 'September'

        # Create datetime objects for begin and end dates
        begin_date_obj = datetime(int(begin_year), int(begin_month), 1)
        end_date_obj = datetime(int(end_year), int(end_month), 1)

    # Adjust the end date by adding one month for period calculation
    adjusted_end_date = end_date_obj + relativedelta(months=1)

    # Calculate the period in years and months
    period = relativedelta(adjusted_end_date, begin_date_obj)
    st.write(f"Period Beginning {begin_month_name} {begin_year} and Ending {end_month_name} {end_year} - {period.years} years and {period.months} months")

    # Filter data between begin_date and end_date, including both
    df_filtered = df[(df['Date'] >= begin_date_obj) & (df['Date'] <= end_date_obj)]

    # Check if the filtering was successful
    if df_filtered.empty:
        st.error("No data available for the selected date range.")
        st.stop()

    # # Display the filtered DataFrame
    # st.subheader("Filtered Data")
    # st.write(df_filtered)

    # Calculate cumulative function for a specific column
    def calculate_cumulative(df, column_name):
        cumulative = [1]  # Start with $1
        for i in range(1, len(df)):
            prev_value = df[column_name].iloc[i-1]
            current_value = df[column_name].iloc[i]

            if prev_value == 0 or pd.isnull(prev_value):
                percentage_change = 0
            else:
                percentage_change = (current_value / prev_value) - 1
            cumulative.append(cumulative[-1] * (1 + percentage_change))
        return cumulative

    # Calculate cumulative values
    df_filtered = df_filtered.reset_index(drop=True)  # Reset index after filtering
    df_filtered['Cumulative Nominal Dividends'] = calculate_cumulative(df_filtered, 'Nominal Dividends')
    df_filtered['Cumulative CPI'] = calculate_cumulative(df_filtered, 'CPI')
    df_filtered['Cumulative Real Dividends'] = calculate_cumulative(df_filtered, 'Real Dividend')
    df_filtered['Cumulative Composite Price'] = calculate_cumulative(df_filtered, 'Composite Price')
    df_filtered['Cumulative Real Price'] = calculate_cumulative(df_filtered, 'Real Price')
    df_filtered['Cumulative Nominal Total Return Price'] = calculate_cumulative(df_filtered, 'Nominal Total Return Price')
    df_filtered['Cumulative Real Total Return Price'] = calculate_cumulative(df_filtered, 'Real Total Return Price')

    # Check if cumulative columns were created successfully
    required_columns = [
        'Cumulative Nominal Dividends',
        'Cumulative CPI',
        'Cumulative Real Dividends',
        'Cumulative Composite Price',
        'Cumulative Real Price',
        'Cumulative Nominal Total Return Price',
        'Cumulative Real Total Return Price'
    ]

    for col in required_columns:
        if col not in df_filtered.columns:
            st.error(f"Error: {col} column was not created.")
            st.stop()

    # Get the final values for all cumulative series
    final_nominal_dividends = df_filtered['Cumulative Nominal Dividends'].iloc[-1]
    final_cpi = df_filtered['Cumulative CPI'].iloc[-1]
    final_real_dividends = df_filtered['Cumulative Real Dividends'].iloc[-1]
    final_composite_price = df_filtered['Cumulative Composite Price'].iloc[-1]
    final_real_price = df_filtered['Cumulative Real Price'].iloc[-1]
    final_nominal_total_return_price = df_filtered['Cumulative Nominal Total Return Price'].iloc[-1]
    final_real_total_return_price = df_filtered['Cumulative Real Total Return Price'].iloc[-1]

    # Function to calculate CAGR
    def calculate_cagr(start_value, end_value, years):
        if start_value > 0 and years > 0:
            return ((end_value / start_value) ** (1 / years) - 1) * 100
        else:
            return None  # Avoid division by zero or negative years

    # Calculate the number of years for CAGR
    years = period.years + period.months / 12.0
    if years == 0:
        years = 0.01  # Prevent division by zero for very short periods

    # Calculate CAGR for relevant columns
    cagr_composite_price = calculate_cagr(1, final_composite_price, years)
    cagr_nominal_total_return = calculate_cagr(1, final_nominal_total_return_price, years)
    cagr_cpi = calculate_cagr(1, final_cpi, years)

    # Display the CAGR values before the nominal data chart
    st.subheader("CAGR Values for Nominal Data")
    st.write(f"CAGR for Composite Price Only: {cagr_composite_price:.2f}%")
    st.write(f"CAGR for Total Return With Dividends: {cagr_nominal_total_return:.2f}%")
    st.write(f"CAGR for CPI: {cagr_cpi:.2f}%")

    # Function to format the result and apply 'lower' logic if below 1
    def format_result(value, label):
        # Calculate the increase percentage
        increase_percent = (value - 1) * 100

        if value < 1:
            percent_lower = (1 - value) * 100
            return f"{label}: (is {percent_lower:.1f}% lower)"
        else:
            return f"{label}: rose from beginning value of 1 to {value:.1f} (Increased by {increase_percent:.1f}%)"

    # Display the final values in Streamlit for Nominal Data
    st.subheader("Final Values for Nominal Data")
    st.write(format_result(final_nominal_dividends, "Dividends"))
    st.write(format_result(final_composite_price, "Composite Price Only"))
    st.write(format_result(final_nominal_total_return_price, "Total Return Including Dividends"))
    st.write(format_result(final_cpi, "CPI"))

    # Create first interactive Plotly chart for nominal data
    fig_nominal = go.Figure()

    # Add Nominal Dividends line
    fig_nominal.add_trace(go.Scatter(x=df_filtered['Date'],
                                     y=df_filtered['Cumulative Nominal Dividends'],
                                     mode='lines',
                                     name='Dividends',
                                     line=dict(color='blue')))

    # Add CPI line
    fig_nominal.add_trace(go.Scatter(x=df_filtered['Date'],
                                     y=df_filtered['Cumulative CPI'],
                                     mode='lines',
                                     name='CPI',
                                     line=dict(color='green')))

    # Add Composite Price line
    fig_nominal.add_trace(go.Scatter(x=df_filtered['Date'],
                                     y=df_filtered['Cumulative Composite Price'],
                                     mode='lines',
                                     name='Composite Price Only',
                                     line=dict(color='purple')))

    # Add Nominal Total Return Price line
    fig_nominal.add_trace(go.Scatter(x=df_filtered['Date'],
                                     y=df_filtered['Cumulative Nominal Total Return Price'],
                                     mode='lines',
                                     name='Total Return Price',
                                     line=dict(color='orange')))

    # Update layout for nominal chart
    fig_nominal.update_layout(
        title="Nominal Ending Values",
        xaxis_title="Date",
        yaxis_title="Cumulative Value",
        hovermode="x unified",
        template="plotly_white"
    )

    # Show nominal data chart
    st.plotly_chart(fig_nominal)

    # Calculate CAGR for Real Price and Real Total Return Price for inflation-adjusted data
    cagr_real_price = calculate_cagr(1, final_real_price, years)
    cagr_real_total_return = calculate_cagr(1, final_real_total_return_price, years)

    # Display the CAGR values before the inflation-adjusted data chart
    st.subheader("CAGR Values for Inflation-Adjusted Data")
    st.write(f"CAGR for Composite Price Only: {cagr_real_price:.2f}%")
    st.write(f"CAGR for Total Return Dividends Reinvested: {cagr_real_total_return:.2f}%")

    # Display the final values in Streamlit for Inflation-Adjusted Data
    st.subheader("Final Values for Inflation-Adjusted Data")
    st.write(format_result(final_real_dividends, "Dividends"))
    st.write(format_result(final_real_price, "Composite Price Only"))
    st.write(format_result(final_real_total_return_price, "Total Return Including Dividends"))

    # Create second interactive Plotly chart for real data
    fig_real = go.Figure()

    # Add Real Dividends line
    fig_real.add_trace(go.Scatter(x=df_filtered['Date'],
                                  y=df_filtered['Cumulative Real Dividends'],
                                  mode='lines',
                                  name='Cumulative Real Dividends',
                                  line=dict(color='red')))

    # Add Real Price line
    fig_real.add_trace(go.Scatter(x=df_filtered['Date'],
                                  y=df_filtered['Cumulative Real Price'],
                                  mode='lines',
                                  name='Cumulative Real Price',
                                  line=dict(color='orange')))

    # Add Real Total Return Price line
    fig_real.add_trace(go.Scatter(x=df_filtered['Date'],
                                  y=df_filtered['Cumulative Real Total Return Price'],
                                  mode='lines',
                                  name='Cumulative Real Total Return Price',
                                  line=dict(color='blue')))

    # Update layout for real chart
    fig_real.update_layout(
        title="Inflation-Adjusted Ending Values",
        xaxis_title="Date",
        yaxis_title="Cumulative Value",
        hovermode="x unified",
        template="plotly_white"
    )

    # Show real data chart
    st.plotly_chart(fig_real)