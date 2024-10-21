import pandas as pd
import numpy as np  # For numerical operations
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

# -----------------------------
# Helper Functions
# -----------------------------

def date_fraction_to_datetime(date_fraction):
    """
    Converts a date fraction to a datetime object.
    """
    # Extract the year
    year = int(date_fraction)

    # Calculate the fractional part representing the month
    fractional_part = date_fraction - year

    # Calculate the month (ensure correct rounding)
    month = int(round(fractional_part * 12 + 0.5))

    # Adjust month and year if necessary
    if month > 12:
        month -= 12
        year += 1
    elif month < 1:
        month = 1

    # Create the datetime object
    date_obj = datetime(year, month, 1)
    return date_obj

def calculate_cagr(start_value, end_value, years):
    """
    Calculates the Compound Annual Growth Rate (CAGR).
    """
    if start_value > 0 and end_value > 0 and years > 0:
        return ((end_value / start_value) ** (1 / years) - 1) * 100
    else:
        return None  # Avoid division by zero or negative years

def display_kpis(label, begin_value, end_value, factor, is_currency=True, decimals=2):
    """
    Displays KPI metrics in three columns.
    """
    # Format the values as currency or numbers
    if is_currency:
        formatted_begin_value = f"${begin_value:,.{decimals}f}"
        formatted_end_value = f"${end_value:,.{decimals}f}"
    else:
        formatted_begin_value = f"{begin_value:,.{decimals}f}"
        formatted_end_value = f"{end_value:,.{decimals}f}"
    formatted_factor = f"{factor:,.{decimals}f}"
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=f"{label} - Begin Value", value=formatted_begin_value)
    with col2:
        st.metric(label=f"{label} - End Value", value=formatted_end_value)
    with col3:
        st.metric(label=f"{label} - Factor Increase", value=formatted_factor)

def create_plot(df, x_col, y_col, title, color, is_currency, decimal_places, yaxis_type):
    """
    Creates a Plotly line chart for the specified data series.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='lines',
        name=y_col.title(),
        line=dict(color=color),
        hovertemplate=f"Date: %{{x|%Y-%m}}<br>Value: {'$' if is_currency else ''}%{{y:,.{decimal_places}f}}<extra></extra>"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        yaxis_type=yaxis_type,  # 'linear' or 'log'
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

# -----------------------------
# Streamlit App Configuration
# -----------------------------

# Title for the Streamlit app
st.title("Standard and Poor's 500 Index Data")

# Sidebar Configuration
st.sidebar.header("Configuration")

# Load your local Excel file directly
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path)
    # Normalize column names by stripping spaces and converting to lowercase
    df.columns = df.columns.str.strip().str.lower()
    # Ensure 'date fraction' column is of type float
    df['date fraction'] = df['date fraction'].astype(float)
    # Convert 'date fraction' to 'date' using the corrected function
    df['date'] = df['date fraction'].apply(date_fraction_to_datetime)
    return df

df = load_data('data.xlsx')

# Define the last valid date based on the availability of data
last_good_date = datetime(2024, 9, 30)  # Set the last usable date to September 30, 2024

# Exclude dates beyond September 2024
df = df[df['date'] <= last_good_date]

# Extract unique years from the 'date' column
years = sorted(df['date'].dt.year.unique())

# Month mapping: Display full names but convert to numeric month strings
month_mapping = {
    'january': '01',
    'february': '02',
    'march': '03',
    'april': '04',
    'may': '05',
    'june': '06',
    'july': '07',
    'august': '08',
    'september': '09',
    'october': '10',
    'november': '11',
    'december': '12'
}

# Initialize 'end_year_default' and 'end_month_default' early to avoid NameError
end_date_default = last_good_date
end_year_default = end_date_default.year
end_month_default = f"{end_date_default.month:02d}"  # September

# Add a dropdown with predefined date ranges
date_options = [
    'Custom Period', 'Last 1 Year', 'Last 2 Years', 'Last 3 Years', 'Last 4 Years',
    'Last 5 Years', 'Last 10 Years', 'Last 15 Years', 'Last 20 Years', 'Last 25 Years',
    'Last 30 Years', 'Last 35 Years', 'Last 40 Years', 'Since WWII'
]
selected_range = st.sidebar.selectbox('Select Time Period', date_options, index=0)  # Default to 'Custom Period'

# Month names list for dropdowns
month_names = list(month_mapping.keys())

# Initialize variables for begin and end dates
begin_year = None
begin_month_name = None
end_year = None
end_month_name = None

# Widgets for selecting begin and end dates (only for custom period)
if selected_range == 'Custom Period':
    # Ensure 1984 is within the range
    if 1984 in years:
        begin_year_index = years.index(1984)
    else:
        begin_year_index = 0  # Default to the first available year
    begin_year = st.sidebar.selectbox('Select the beginning year', options=years, index=begin_year_index)
    begin_month_name = st.sidebar.selectbox('Select the beginning month', options=month_names, index=9)  # October is at index 9

    # Ensure the end year selection doesn't exceed available years
    if end_year_default in years:
        end_year_index = years.index(end_year_default)
    else:
        end_year_index = -1  # Default to the last available year
    end_year = st.sidebar.selectbox('Select the ending year', options=years, index=end_year_index)
    end_month_name = st.sidebar.selectbox('Select the ending month', options=month_names, index=8)  # September is at index 8

# Slider for beginning value for Total Return Including Dividends
begin_total_return_value = st.sidebar.slider(
    "Select Beginning Value for Total Return Including Dividends",
    min_value=10000,
    max_value=1000000,
    value=100000,
    step=10000
)

# Slider for selecting the number of decimal places
decimal_places = st.sidebar.slider(
    "Select Number of Decimal Places",
    min_value=0,
    max_value=4,
    value=2,
    step=1
)

# Determine begin and end dates based on the selection
if selected_range != 'Custom Period':
    # Fixed end date
    end_date_obj = end_date_default

    # Determine the begin date based on the selection
    if selected_range == 'Last 1 Year':
        begin_date_obj = end_date_obj - relativedelta(years=1) + relativedelta(months=1)
    elif selected_range == 'Last 2 Years':
        begin_date_obj = end_date_obj - relativedelta(years=2) + relativedelta(months=1)
    elif selected_range == 'Last 3 Years':
        begin_date_obj = end_date_obj - relativedelta(years=3) + relativedelta(months=1)
    elif selected_range == 'Last 4 Years':
        begin_date_obj = end_date_obj - relativedelta(years=4) + relativedelta(months=1)
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
    st.sidebar.write(f"**Begin Date:** {begin_month_name.capitalize()} {begin_year}")
    st.sidebar.write(f"**End Date:** {end_month_name.capitalize()} {end_year}")

else:
    # Custom Period: Use the values from the widgets
    # Convert month names to the numeric format
    begin_month = month_mapping[begin_month_name]
    end_month = month_mapping[end_month_name]

    # Ensure that end date does not go beyond September 30, 2024
    if int(end_year) > end_year_default or (int(end_year) == end_year_default and int(end_month) > int(end_month_default)):
        st.warning(f"The last usable date is September 30, 2024. Adjusting end date to September 30, 2024.")
        end_year = end_year_default
        end_month = end_month_default
        end_month_name = 'september'

    # Create datetime objects for begin and end dates
    try:
        begin_date_obj = datetime(int(begin_year), int(begin_month), 1)
    except ValueError:
        st.error("Invalid beginning date selected.")
        st.stop()
    try:
        end_date_obj = datetime(int(end_year), int(end_month), 1)
    except ValueError:
        st.error("Invalid ending date selected.")
        st.stop()

# Adjust the end date for period calculation
adjusted_end_date = end_date_obj + relativedelta(months=1)

# Calculate the period in years and months using the adjusted end date
period = relativedelta(adjusted_end_date, begin_date_obj)

# Display the period using the correct end date
st.write(f"**Period:** Beginning {begin_month_name.capitalize()} {begin_year} and Ending {end_month_name.capitalize()} {end_year} - {period.years} years and {period.months} months")

# Filter data between begin_date and end_date, including both
df_filtered = df[(df['date'] >= begin_date_obj) & (df['date'] <= end_date_obj)]

# Check if the filtering was successful
if df_filtered.empty:
    st.error("No data available for the selected date range.")
    st.stop()

# Reset index after filtering
df_filtered = df_filtered.reset_index(drop=True)

# Define required columns with normalized names
required_columns = [
    'nominal dividends',
    'cpi',
    'real dividend',
    'composite price',
    'real price',
    'nominal total return price',
    'real total return price',
    'nominal earnings',
    'real earnings'
]

# Verify that all required columns exist
missing_columns = [col for col in required_columns if col not in df_filtered.columns]
if missing_columns:
    st.error(f"The following required columns are missing from the data: {', '.join(missing_columns)}")
    st.stop()

# Calculate the number of years for CAGR
years_cagr = period.years + period.months / 12.0
if years_cagr <= 0:
    st.error("The selected period is too short to calculate CAGR.")
    st.stop()

# Calculate CAGR for relevant columns using actual data
cagr_composite_price = calculate_cagr(df_filtered['composite price'].iloc[0], df_filtered['composite price'].iloc[-1], years_cagr)
cagr_nominal_total_return = calculate_cagr(df_filtered['nominal total return price'].iloc[0], df_filtered['nominal total return price'].iloc[-1], years_cagr)
cagr_cpi = calculate_cagr(df_filtered['cpi'].iloc[0], df_filtered['cpi'].iloc[-1], years_cagr)
cagr_real_price = calculate_cagr(df_filtered['real price'].iloc[0], df_filtered['real price'].iloc[-1], years_cagr)
cagr_real_total_return = calculate_cagr(df_filtered['real total return price'].iloc[0], df_filtered['real total return price'].iloc[-1], years_cagr)

# Display the CAGR values before the nominal data chart
st.subheader("CAGR Values for Nominal Data")
st.write(f"CAGR for Composite Price: {cagr_composite_price:.{decimal_places}f}%")
st.write(f"CAGR for Total Return With Dividends: {cagr_nominal_total_return:.{decimal_places}f}%")
st.write(f"CAGR for CPI: {cagr_cpi:.{decimal_places}f}%")

# Extract beginning and ending values for each series from actual data columns
data_columns = [
    'nominal dividends',
    'cpi',
    'real dividend',
    'composite price',
    'real price',
    'nominal total return price',
    'real total return price',
    'nominal earnings',
    'real earnings'
]

begin_end_values = {}
for col in data_columns:
    try:
        begin_val = df_filtered[col].iloc[0]
        end_val = df_filtered[col].iloc[-1]
        begin_end_values[col] = (begin_val, end_val)
    except KeyError:
        st.error(f"The column '{col}' does not exist in the data.")
        st.stop()

# Calculate the factor increase for each series
factors = {}
for col in data_columns:
    begin_val = begin_end_values[col][0]
    end_val = begin_end_values[col][1]
    if begin_val != 0:
        factor = end_val / begin_val
    else:
        factor = None  # Avoid division by zero
    factors[col] = factor

# Calculate the ending value based on CAGR and the selected beginning value from the slider
ending_value_cagr = begin_total_return_value * ((1 + (cagr_nominal_total_return / 100)) ** years_cagr)

# Calculate the factor increase for Total Return Including Dividends
factor_total_return = ending_value_cagr / begin_total_return_value

# -----------------------------
# Selection Widgets for Charts
# -----------------------------

st.sidebar.header("Chart Selections")

# Define available data series for nominal and real charts (normalized)
nominal_series = {
    'nominal earnings': {'color': 'red', 'is_currency': True},
    'nominal dividends': {'color': 'blue', 'is_currency': True},
    'cpi': {'color': 'green', 'is_currency': False},
    'composite price': {'color': 'purple', 'is_currency': False},
    'nominal total return price': {'color': 'orange', 'is_currency': True}  # Set is_currency to True
}

real_series = {
    'real earnings': {'color': 'green', 'is_currency': True},
    'real dividend': {'color': 'red', 'is_currency': True},
    'real price': {'color': 'orange', 'is_currency': False},
    'real total return price': {'color': 'blue', 'is_currency': True}  # Set is_currency to True
}

# Selection widgets (using normalized names)
selected_nominal = st.sidebar.radio(
    "Select Nominal Data Series to Display",
    options=list(nominal_series.keys()),
    index=0
)

selected_real = st.sidebar.radio(
    "Select Inflation-Adjusted (Real) Data Series to Display",
    options=list(real_series.keys()),
    index=0
)

# -----------------------------
# Display KPI Metrics
# -----------------------------

# Display KPIs for Nominal Data
with st.expander("Nominal Data KPIs"):
    if selected_nominal == 'nominal total return price':
        # Use slider value as beginning value and calculate ending value using CAGR
        begin_value = begin_total_return_value
        end_value = ending_value_cagr
        factor = factor_total_return
    else:
        # Use actual data for other series
        begin_value = begin_end_values[selected_nominal][0]
        end_value = begin_end_values[selected_nominal][1]
        factor = factors[selected_nominal]

    display_kpis(
        label=selected_nominal.title(),
        begin_value=begin_value,
        end_value=end_value,
        factor=factor,
        is_currency=nominal_series[selected_nominal]['is_currency'],
        decimals=decimal_places
    )

# Display the final values in Streamlit for Nominal Data
if selected_nominal == 'nominal total return price':
    st.subheader("Final Values for Nominal Total Return Price")
    st.write(f"Beginning Value: ${begin_value:,.{decimal_places}f}")
    st.write(f"Ending Value: ${end_value:,.{decimal_places}f}")
    st.write(f"Factor Increase: {factor:,.{decimal_places}f}")

# -----------------------------
# Plotting Charts
# -----------------------------

# Adjust the data series for plotting if needed
if selected_nominal == 'nominal total return price':
    # Calculate the scaling factor
    scaling_factor_nominal = begin_total_return_value / df_filtered[selected_nominal].iloc[0]
    # Create the adjusted data series
    df_filtered['adjusted_nominal_series'] = df_filtered[selected_nominal] * scaling_factor_nominal
    # Use the adjusted data series for plotting
    y_data_nominal = 'adjusted_nominal_series'
else:
    # Use the actual data
    y_data_nominal = selected_nominal

# Create Nominal Data Chart
fig_nominal_selected = create_plot(
    df_filtered,
    x_col='date',
    y_col=y_data_nominal,
    title=f"{selected_nominal.title()}",
    color=nominal_series[selected_nominal]['color'],
    is_currency=nominal_series[selected_nominal]['is_currency'],
    decimal_places=decimal_places,
    yaxis_type="linear"  # Linear scale
)

st.plotly_chart(fig_nominal_selected, use_container_width=True)

# -----------------------------
# Inflation-Adjusted (Real) Data Section
# -----------------------------

# Display the CAGR values before the inflation-adjusted data chart
st.subheader("CAGR Values for Inflation-Adjusted Data")
st.write(f"CAGR for Real Price: {cagr_real_price:.{decimal_places}f}%")
st.write(f"CAGR for Real Total Return Dividends Reinvested: {cagr_real_total_return:.{decimal_places}f}%")

# Calculate the ending value for inflation-adjusted total return, using the slider value
ending_value_cagr_real = begin_total_return_value * ((1 + (cagr_real_total_return / 100)) ** years_cagr)

# Calculate the factor increase for inflation-adjusted total return
factor_real_total_return = ending_value_cagr_real / begin_total_return_value

# Display KPIs for Real Data
with st.expander("Inflation-Adjusted (Real) Data KPIs"):
    if selected_real == 'real total return price':
        # Use slider value as beginning value and calculate ending value using CAGR
        begin_value_real = begin_total_return_value
        end_value_real = ending_value_cagr_real
        factor_real = factor_real_total_return
    else:
        # Use actual data for other series
        begin_value_real = begin_end_values[selected_real][0]
        end_value_real = begin_end_values[selected_real][1]
        factor_real = factors[selected_real]

    display_kpis(
        label=selected_real.title(),
        begin_value=begin_value_real,
        end_value=end_value_real,
        factor=factor_real,
        is_currency=real_series[selected_real]['is_currency'],
        decimals=decimal_places
    )

# Display the final values in Streamlit for Real Data
if selected_real == 'real total return price':
    st.subheader("Final Values for Real Total Return Price")
    st.write(f"Beginning Value: ${begin_value_real:,.{decimal_places}f}")
    st.write(f"Ending Value: ${end_value_real:,.{decimal_places}f}")
    st.write(f"Factor Increase: {factor_real:,.{decimal_places}f}")

# Adjust the data series for plotting if needed
if selected_real == 'real total return price':
    # Calculate the scaling factor
    scaling_factor_real = begin_total_return_value / df_filtered[selected_real].iloc[0]
    # Create the adjusted data series
    df_filtered['adjusted_real_series'] = df_filtered[selected_real] * scaling_factor_real
    # Use the adjusted data series for plotting
    y_data_real = 'adjusted_real_series'
else:
    # Use the actual data
    y_data_real = selected_real

# Create Real Data Chart
fig_real_selected = create_plot(
    df_filtered,
    x_col='date',
    y_col=y_data_real,
    title=f"{selected_real.title()}",
    color=real_series[selected_real]['color'],
    is_currency=real_series[selected_real]['is_currency'],
    decimal_places=decimal_places,
    yaxis_type="linear"  # Linear scale
)

st.plotly_chart(fig_real_selected, use_container_width=True)