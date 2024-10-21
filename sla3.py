import pandas as pd
import numpy as np  # For numerical operations
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
from io import BytesIO  # For creating in-memory Excel files

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

def display_kpis(label, begin_value, end_value, factor, is_currency=True, decimals_begin_end=2):
    """
    Displays KPI metrics in three columns with specified formatting.
    """
    # Format the values as currency or numbers
    if is_currency:
        formatted_begin_value = f"${begin_value:,.{decimals_begin_end}f}"
        formatted_end_value = f"${end_value:,.{decimals_begin_end}f}"
    else:
        formatted_begin_value = f"{begin_value:,.{decimals_begin_end}f}"
        formatted_end_value = f"{end_value:,.{decimals_begin_end}f}"

    formatted_factor = f"{factor:,.2f}"  # Always two decimal places for factor increase

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

# Add a dropdown with predefined date ranges, including up to "Last 100 Years"
date_options = [
    'Custom Period',
    'Last 1 Year', 'Last 2 Years', 'Last 3 Years', 'Last 4 Years',
    'Last 5 Years', 'Last 10 Years', 'Last 15 Years', 'Last 20 Years',
    'Last 25 Years', 'Last 30 Years', 'Last 35 Years', 'Last 40 Years',
    'Last 50 Years', 'Last 60 Years', 'Last 70 Years', 'Last 80 Years',
    'Last 90 Years', 'Last 100 Years',
    'Since WWII'
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

# Removed the decimal slider
# Set a fixed number of decimal places
decimal_places = 2

# Determine begin and end dates based on the selection
if selected_range.startswith('Last') and selected_range.endswith('Years'):
    try:
        # Extract the number of years from the selected option
        n_years = int(selected_range.split()[1])
        # Calculate the begin date
        begin_date_obj = end_date_obj = end_date_default  # Initialize end_date_obj
        begin_date_obj = end_date_obj - relativedelta(years=n_years) + relativedelta(months=1)
    except (IndexError, ValueError):
        st.error("Invalid selection for the number of years.")
        st.stop()
elif selected_range == 'Since WWII':
    begin_date_obj = datetime(1945, 10, 1)  # Start from October 1945
elif selected_range == 'Custom Period':
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
else:
    st.error("Invalid time period selected.")
    st.stop()

if selected_range.startswith('Last') and selected_range.endswith('Years'):
    # Display the calculated dates in the sidebar
    st.sidebar.write(f"**Begin Date:** {begin_date_obj.strftime('%B %Y')}")
    st.sidebar.write(f"**End Date:** {end_date_default.strftime('%B %Y')}")
elif selected_range == 'Since WWII':
    st.sidebar.write(f"**Begin Date:** {begin_date_obj.strftime('%B %Y')}")
    st.sidebar.write(f"**End Date:** {end_date_default.strftime('%B %Y')}")
elif selected_range == 'Custom Period':
    # Calculate the period in years and months
    adjusted_end_date = end_date_obj + relativedelta(months=1)
    period = relativedelta(adjusted_end_date, begin_date_obj)
    # Display the period
    st.write(f"**Period:** Beginning {begin_date_obj.strftime('%B %Y')} and Ending {end_date_obj.strftime('%B %Y')} - {period.years} years and {period.months} months")

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
    'real dividends',
    'composite price only',
    'real composite price only',
    'total return',
    'real total return',
    'nominal earnings',
    'real earnings'
]

# Verify that all required columns exist
missing_columns = [col for col in required_columns if col not in df_filtered.columns]
if missing_columns:
    st.error(f"The following required columns are missing from the data: {', '.join(missing_columns)}")
    st.stop()

# Calculate the number of years for CAGR
if selected_range.startswith('Last') and selected_range.endswith('Years'):
    years_cagr = n_years + 0  # Assuming full years; adjust if partial months are needed
elif selected_range == 'Since WWII':
    period = relativedelta(end_date_default, begin_date_obj)
    years_cagr = period.years + period.months / 12.0
elif selected_range == 'Custom Period':
    period = relativedelta(end_date_obj + relativedelta(months=1), begin_date_obj)
    years_cagr = period.years + period.months / 12.0

if years_cagr <= 0:
    st.error("The selected period is too short to calculate CAGR.")
    st.stop()

# Calculate CAGR for relevant columns using actual data
cagr_composite_price = calculate_cagr(df_filtered['composite price only'].iloc[0], df_filtered['composite price only'].iloc[-1], years_cagr)
cagr_nominal_total_return = calculate_cagr(df_filtered['total return'].iloc[0], df_filtered['total return'].iloc[-1], years_cagr)
cagr_cpi = calculate_cagr(df_filtered['cpi'].iloc[0], df_filtered['cpi'].iloc[-1], years_cagr)
cagr_real_price = calculate_cagr(df_filtered['real composite price only'].iloc[0], df_filtered['real composite price only'].iloc[-1], years_cagr)
cagr_real_total_return = calculate_cagr(df_filtered['real total return'].iloc[0], df_filtered['real total return'].iloc[-1], years_cagr)

# Display the CAGR values before the nominal data chart
st.subheader("CAGR Values for Nominal Data")
if cagr_composite_price is not None:
    st.write(f"CAGR for Composite Price Only: {cagr_composite_price:.2f}%")
else:
    st.write("CAGR for Composite Price Only: N/A")

if cagr_nominal_total_return is not None:
    st.write(f"CAGR for Total Return With Dividends: {cagr_nominal_total_return:.2f}%")
else:
    st.write("CAGR for Total Return With Dividends: N/A")

if cagr_cpi is not None:
    st.write(f"CAGR for CPI: {cagr_cpi:.2f}%")
else:
    st.write("CAGR for CPI: N/A")

# Extract beginning and ending values for each series from actual data columns
data_columns = [
    'nominal dividends',
    'cpi',
    'real dividends',
    'composite price only',
    'real composite price only',
    'total return',
    'real total return',
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
ending_value_cagr = begin_total_return_value * ((1 + (cagr_nominal_total_return / 100)) ** years_cagr) if cagr_nominal_total_return is not None else None

# Calculate the factor increase for Total Return Including Dividends
factor_total_return = ending_value_cagr / begin_total_return_value if ending_value_cagr is not None else None

# -----------------------------
# Selection Widgets for Charts
# -----------------------------

st.sidebar.header("Chart Selections")

# Define available data series for nominal and real charts (normalized)
nominal_series = {
    'nominal earnings': {'color': 'red', 'is_currency': True},
    'nominal dividends': {'color': 'blue', 'is_currency': True},
    'cpi': {'color': 'green', 'is_currency': False},
    'composite price only': {'color': 'purple', 'is_currency': False},
    'total return': {'color': 'orange', 'is_currency': True}
}

real_series = {
    'real earnings': {'color': 'green', 'is_currency': True},
    'real dividends': {'color': 'red', 'is_currency': True},
    'real composite price only': {'color': 'orange', 'is_currency': False},
    'real total return': {'color': 'blue', 'is_currency': True}  # Added Real Total Return
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
# Display KPI Metrics for Nominal Data
# -----------------------------

# Initialize begin_value, end_value, and factor for nominal data
if selected_nominal == 'total return':
    # Use slider value as beginning value and calculate ending value using CAGR
    begin_value = begin_total_return_value
    end_value = ending_value_cagr if ending_value_cagr is not None else 0
    factor = factor_total_return if factor_total_return is not None else 0
    decimals_begin_end = 0  # Zero decimal places for begin and end values
else:
    # Use actual data for other series
    begin_value = begin_end_values[selected_nominal][0]
    end_value = begin_end_values[selected_nominal][1]
    factor = factors[selected_nominal] if factors[selected_nominal] is not None else 0
    decimals_begin_end = 2  # Two decimal places for begin and end values

display_kpis(
    label=selected_nominal.title(),
    begin_value=begin_value,
    end_value=end_value,
    factor=factor,
    is_currency=nominal_series[selected_nominal]['is_currency'],
    decimals_begin_end=decimals_begin_end
)

# -----------------------------
# Plotting Nominal Data Chart
# -----------------------------

# Adjust the data series for plotting if needed
if selected_nominal == 'total return' and ending_value_cagr is not None:
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
    decimal_places=decimal_places,  # Use the fixed decimal places
    yaxis_type="linear"  # Linear scale
)

st.plotly_chart(fig_nominal_selected, use_container_width=True)

# -----------------------------
# Display KPI Metrics for Real Data
# -----------------------------

# Calculate the ending value for inflation-adjusted total return, using the fixed value
if cagr_real_total_return is not None:
    ending_value_cagr_real = begin_total_return_value * ((1 + (cagr_real_total_return / 100)) ** years_cagr)
    # Calculate the factor increase for inflation-adjusted total return
    factor_real_total_return = ending_value_cagr_real / begin_total_return_value
else:
    ending_value_cagr_real = None
    factor_real_total_return = None

# Initialize begin_value, end_value, and factor for real data
if selected_real == 'real total return':
    # Use fixed value as beginning value and calculate ending value using CAGR
    begin_value_real = begin_total_return_value
    end_value_real = ending_value_cagr_real if ending_value_cagr_real is not None else 0
    factor_real = factor_real_total_return if factor_real_total_return is not None else 0
    decimals_begin_end_real = 0  # Zero decimal places for begin and end values
else:
    # Use actual data for other series
    begin_value_real = begin_end_values[selected_real][0]
    end_value_real = begin_end_values[selected_real][1]
    factor_real = factors[selected_real] if factors[selected_real] is not None else 0
    decimals_begin_end_real = 2  # Two decimal places for begin and end values

display_kpis(
    label=selected_real.title(),
    begin_value=begin_value_real,
    end_value=end_value_real,
    factor=factor_real,
    is_currency=real_series[selected_real]['is_currency'],
    decimals_begin_end=decimals_begin_end_real
)

# -----------------------------
# Plotting Real Data Chart
# -----------------------------

# Adjust the data series for plotting if needed
if selected_real == 'real total return' and ending_value_cagr_real is not None:
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
    decimal_places=decimal_places,  # Use the fixed decimal places
    yaxis_type="linear"  # Linear scale
)

st.plotly_chart(fig_real_selected, use_container_width=True)

# -----------------------------
# Create Metrics DataFrame
# -----------------------------

# Initialize a list to store metrics data
metrics_data = []

# Define metric display names and corresponding column names
nominal_metrics = [
    ('Nominal Earnings', 'nominal earnings'),
    ('Nominal Dividends', 'nominal dividends'),
    ('Composite Price Only', 'composite price only'),
    ('Total Return', 'total return'),
    ('CPI', 'cpi')
]

real_metrics = [
    ('Real Earnings', 'real earnings'),
    ('Real Dividends', 'real dividends'),
    ('Real Composite Price Only', 'real composite price only'),
    ('Real Total Return', 'real total return')  # Added Real Total Return
]

# Mapping for is_currency
metric_is_currency = {
    'Nominal Earnings': True,
    'Nominal Dividends': True,
    'Composite Price Only': False,
    'Total Return': True,
    'CPI': False,
    'Real Earnings': True,
    'Real Dividends': True,
    'Real Composite Price Only': False,
    'Real Total Return': True  # Assuming Real Total Return is a currency
}

# Collect nominal metrics
for display_name, col_name in nominal_metrics:
    if display_name == 'Total Return':
        # Use adjusted values based on fixed begin_total_return_value
        begin_val = begin_total_return_value
        end_val = ending_value_cagr if ending_value_cagr is not None else 0
        factor = factor_total_return if factor_total_return is not None else 0
    else:
        begin_val, end_val = begin_end_values[col_name]
        factor = factors[col_name] if factors[col_name] is not None else 0
    metrics_data.append({
        'Metric': display_name,
        'Begin Value': begin_val,
        'End Value': end_val,
        'Factor Increase': factor
    })

# Add a blank line for spacing
metrics_data.append({'Metric': '', 'Begin Value': '', 'End Value': '', 'Factor Increase': ''})

# Collect real metrics
for display_name, col_name in real_metrics:
    if display_name == 'Real Total Return':
        # Use adjusted values based on CAGR
        begin_val = begin_total_return_value
        end_val = ending_value_cagr_real if ending_value_cagr_real is not None else 0
        factor = factor_real_total_return if factor_real_total_return is not None else 0
    else:
        begin_val, end_val = begin_end_values[col_name]
        factor = factors[col_name] if factors[col_name] is not None else 0
    metrics_data.append({
        'Metric': display_name,
        'Begin Value': begin_val,
        'End Value': end_val,
        'Factor Increase': factor
    })

# Add a blank line before CAGR metrics
metrics_data.append({'Metric': '', 'Begin Value': '', 'End Value': '', 'Factor Increase': ''})

# Add CAGR metrics by placing CAGR values in the Begin Value column
cagr_metrics = [
    ('Nominal CAGR Price Only', cagr_composite_price),
    ('Nominal CAGR Dividends Reinvested', cagr_nominal_total_return),
    ('Real CAGR Price Only', cagr_real_price),
    ('Real CAGR Dividends Reinvested', cagr_real_total_return)
]

for i, (metric_name, cagr_value) in enumerate(cagr_metrics):
    metrics_data.append({
        'Metric': metric_name,
        'Begin Value': cagr_value,  # Keep as float for proper formatting
        'End Value': '',
        'Factor Increase': ''
    })
    # Insert a blank row after the second CAGR metric
    if i == 1:
        metrics_data.append({'Metric': '', 'Begin Value': '', 'End Value': '', 'Factor Increase': ''})

# Create DataFrame from metrics data
metrics_df = pd.DataFrame(metrics_data)

# -----------------------------
# Format and Display Metrics Table
# -----------------------------

# Function to format values (excluding CAGR)
def format_value_no_cagr(value, is_currency):
    if value == '' or pd.isna(value):
        return ''
    if is_currency:
        return f"${value:,.0f}"  # No decimals for Total Return
    else:
        return f"{value:,.2f}"

# Create a formatted DataFrame for display
formatted_metrics_data = []
for index, row in metrics_df.iterrows():
    metric_name = row['Metric']
    begin_value = row['Begin Value']
    end_value = row['End Value']
    factor_increase = row['Factor Increase']

    # Skip formatting for blank lines
    if metric_name == '':
        formatted_metrics_data.append(row)
        continue

    # Determine if the metric should be formatted as currency
    is_currency = metric_is_currency.get(metric_name, False)

    # Check if the row is a CAGR metric
    if 'CAGR' in metric_name:
        # For CAGR metrics, format as percentage with two decimals
        if pd.notna(begin_value):
            formatted_begin_value = f"{begin_value:.2f}%"
        else:
            formatted_begin_value = 'N/A'
        # No formatting for End Value and Factor Increase
        formatted_end_value = ''
        formatted_factor_increase = ''
    else:
        # Format Begin Value and End Value
        if metric_name in ['Total Return', 'Real Total Return']:
            # No decimals, currency formatting
            formatted_begin_value = format_value_no_cagr(begin_value, is_currency)
            formatted_end_value = format_value_no_cagr(end_value, is_currency)
        else:
            # Two decimals
            formatted_begin_value = f"{begin_value:,.2f}"
            formatted_end_value = f"{end_value:,.2f}"

        # Format Factor Increase
        if factor_increase == '' or pd.isna(factor_increase):
            formatted_factor_increase = ''
        else:
            formatted_factor_increase = f"{factor_increase:,.2f}"

    formatted_metrics_data.append({
        'Metric': metric_name,
        'Begin Value': formatted_begin_value,
        'End Value': formatted_end_value,
        'Factor Increase': formatted_factor_increase
    })

# Create a DataFrame for display
formatted_metrics_df = pd.DataFrame(formatted_metrics_data)

# Display the metrics table using st.table to avoid scrolling
st.subheader("Metrics Summary")
st.table(formatted_metrics_df)

# -----------------------------
# Download Button for Metrics
# -----------------------------

# Function to convert DataFrame to Excel with custom formatting
@st.cache_data
def convert_df_to_excel(formatted_df, original_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        formatted_df.to_excel(writer, index=False, sheet_name='Metrics')

        workbook = writer.book
        worksheet = writer.sheets['Metrics']

        # Define formats with center alignment
        currency_no_decimals_centered_format = workbook.add_format({'num_format': '$#,##0', 'align': 'center'})
        number_two_decimals_centered_format = workbook.add_format({'num_format': '#,##0.00', 'align': 'center'})
        percentage_two_decimals_centered_format = workbook.add_format({'num_format': '0.00%', 'align': 'center'})
        general_centered_format = workbook.add_format({'align': 'center'})

        # Get the number of rows and columns
        max_row, max_col = formatted_df.shape

        for row in range(max_row):
            metric = original_df.at[row, 'Metric']
            if pd.isna(metric) or metric == '':
                continue  # Skip formatting for blank lines
            elif 'CAGR' in metric:
                # Apply percentage format to Begin Value column (Column B, index 1)
                cell_value = original_df.at[row, 'Begin Value']
                if isinstance(cell_value, float) or isinstance(cell_value, int):
                    # Write as number divided by 100 for percentage formatting
                    worksheet.write_number(row + 1, 1, cell_value / 100, percentage_two_decimals_centered_format)
                else:
                    # Write as string (e.g., 'N/A')
                    worksheet.write(row + 1, 1, cell_value, general_centered_format)
                # Leave End Value and Factor Increase empty
            elif metric in ['Total Return', 'Real Total Return']:
                # Apply currency format with no decimals to Begin Value and End Value
                # Begin Value (Column B, index 1)
                begin_val_str = formatted_df.at[row, 'Begin Value'].replace('$', '').replace(',', '')
                try:
                    begin_val_num = float(begin_val_str) if begin_val_str else 0.0
                except ValueError:
                    begin_val_num = 0.0  # Default to 0.0 if conversion fails
                worksheet.write_number(row + 1, 1, begin_val_num, currency_no_decimals_centered_format)

                # End Value (Column C, index 2)
                end_val_str = formatted_df.at[row, 'End Value'].replace('$', '').replace(',', '')
                try:
                    end_val_num = float(end_val_str) if end_val_str else 0.0
                except ValueError:
                    end_val_num = 0.0  # Default to 0.0 if conversion fails
                worksheet.write_number(row + 1, 2, end_val_num, currency_no_decimals_centered_format)

                # Factor Increase remains as is (string or number)
                factor_increase = original_df.at[row, 'Factor Increase']
                if isinstance(factor_increase, (float, int)):
                    worksheet.write_number(row + 1, 3, factor_increase, number_two_decimals_centered_format)
                else:
                    worksheet.write(row + 1, 3, factor_increase, general_centered_format)
            else:
                # Apply number format with two decimals to Begin Value and End Value
                # Begin Value (Column B, index 1)
                begin_val = original_df.at[row, 'Begin Value']
                if isinstance(begin_val, (float, int)):
                    worksheet.write_number(row + 1, 1, begin_val, number_two_decimals_centered_format)
                else:
                    worksheet.write(row + 1, 1, begin_val, general_centered_format)

                # End Value (Column C, index 2)
                end_val = original_df.at[row, 'End Value']
                if isinstance(end_val, (float, int)):
                    worksheet.write_number(row + 1, 2, end_val, number_two_decimals_centered_format)
                else:
                    worksheet.write(row + 1, 2, end_val, general_centered_format)

                # Factor Increase (Column D, index 3)
                factor_increase = original_df.at[row, 'Factor Increase']
                if isinstance(factor_increase, (float, int)):
                    worksheet.write_number(row + 1, 3, factor_increase, number_two_decimals_centered_format)
                else:
                    worksheet.write(row + 1, 3, factor_increase, general_centered_format)

        # Adjust column widths for better readability
        worksheet.set_column('A:A', 35, general_centered_format)  # Metric
        worksheet.set_column('B:B', 20, general_centered_format)  # Begin Value
        worksheet.set_column('C:C', 20, general_centered_format)  # End Value
        worksheet.set_column('D:D', 18, general_centered_format)  # Factor Increase

    # Convert the metrics DataFrame to Excel with formatting
    excel_data = convert_df_to_excel(formatted_metrics_df, metrics_df)

    # Provide a download button
    st.download_button(
        label="Download Metrics as Excel",
        data=excel_data,
        file_name='metrics_summary.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )