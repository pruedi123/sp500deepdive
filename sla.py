import pandas as pd
import streamlit as st

# Title for the Streamlit app
st.title("Standard and Poors 500 Index Data")

# Load your local Excel file directly (without upload)
df = pd.read_excel('data.xlsx')

# Extract unique years from the 'Date' column
years = sorted(list(set(df['Date'].apply(lambda x: int(x)))))  # Extract years as integers

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

# Sidebar for date selection and investment slider
st.sidebar.header("Select Dates and Initial Investment")

# Dropdowns for selecting begin year and month, with default values set
begin_year = st.sidebar.selectbox('Select the beginning year', options=years, index=years.index(1984))
begin_month_name = st.sidebar.selectbox('Select the beginning month', options=list(month_mapping.keys()), index=7)  # August is at index 7

# Dropdowns for selecting end year and month, with default values set
end_year = st.sidebar.selectbox('Select the ending year', options=years, index=years.index(2024))
end_month_name = st.sidebar.selectbox('Select the ending month', options=list(month_mapping.keys()), index=8)  # September is at index 8

# Convert month names to the numeric format
begin_month = month_mapping[begin_month_name]
end_month = month_mapping[end_month_name]

# Combine year and month to create the date format (e.g., 2000.01 for January 2000)
begin_date = float(f"{begin_year}.{begin_month}")
end_date = float(f"{end_year}.{end_month}")

# Slider for initial investment
initial_investment = st.sidebar.slider('Select initial investment ($)', min_value=0, max_value=1000000, value=100000, step=1000)

# Function to calculate the number of years between begin and end dates
def calculate_years(df, begin_date, end_date):
    begin_index = df[df['Date'] == begin_date].index[0]
    end_index = df[df['Date'] == end_date].index[0]
    return (end_index - begin_index) / 12  # Assuming monthly data

# Function to calculate CAGR
def cagr(begin_value, end_value, years):
    return ((end_value / begin_value) ** (1 / years) - 1) * 100

# Function to calculate Nominal Return CAGR
def nominal_return_cagr(df, begin_date, end_date, years):
    begin_price = df.loc[df['Date'] == begin_date, 'Composite Price'].values[0]
    end_price = df.loc[df['Date'] == end_date, 'Composite Price'].values[0]
    return round(cagr(begin_price, end_price, years), 2)

# Function to calculate Total Return with Dividend Reinvested CAGR
def total_return_cagr(df, begin_date, end_date, years):
    begin_total_return_price = df.loc[df['Date'] == begin_date, 'Nominal Total Return Price'].values[0]
    end_total_return_price = df.loc[df['Date'] == end_date, 'Nominal Total Return Price'].values[0]
    return round(cagr(begin_total_return_price, end_total_return_price, years), 2)

# Function to calculate Real Return No Dividend CAGR
def real_return_no_div_cagr(df, begin_date, end_date, years):
    begin_real_price = df.loc[df['Date'] == begin_date, 'Real Price'].values[0]
    end_real_price = df.loc[df['Date'] == end_date, 'Real Price'].values[0]
    return round(cagr(begin_real_price, end_real_price, years), 2)

# Function to calculate Real Return With Dividend Reinvestment CAGR
def real_return_with_div_cagr(df, begin_date, end_date, years):
    begin_real_total_return_price = df.loc[df['Date'] == begin_date, 'Real Total Return Price'].values[0]
    end_real_total_return_price = df.loc[df['Date'] == end_date, 'Real Total Return Price'].values[0]
    return round(cagr(begin_real_total_return_price, end_real_total_return_price, years), 2)

# Function to calculate Inflation Rate CAGR
def inflation_cagr(df, begin_date, end_date, years):
    begin_cpi = df.loc[df['Date'] == begin_date, 'CPI'].values[0]
    end_cpi = df.loc[df['Date'] == end_date, 'CPI'].values[0]
    return begin_cpi, end_cpi, round(cagr(begin_cpi, end_cpi, years), 2)

# Function to calculate Dividend CAGR
def dividend_cagr(df, begin_date, end_date, years):
    begin_dividend = df.loc[df['Date'] == begin_date, 'Nominal Dividends'].values[0]
    end_dividend = df.loc[df['Date'] == end_date, 'Nominal Dividends'].values[0]
    return begin_dividend, end_dividend, round(cagr(begin_dividend, end_dividend, years), 2)

# Function to calculate the ending value based on CAGR
def calculate_ending_value(initial_investment, cagr, years):
    return round(initial_investment * (1 + cagr / 100) ** years, 0)

# Function to calculate the Composite Price summary (beginning, ending, and ratio)
def composite_price_summary(df, begin_date, end_date):
    # Get the beginning and ending values for Composite Price
    begin_value = df.loc[df['Date'] == begin_date, 'Composite Price'].values[0]
    end_value = df.loc[df['Date'] == end_date, 'Composite Price'].values[0]
    
    # Calculate the ratio of ending value to beginning value
    ratio = end_value / begin_value
    
    return begin_value, end_value, ratio

# Calculate results and display them in Streamlit
if st.button('Calculate CAGR, Dividends, and Ending Values'):
    # Calculate the number of years between the begin and end date
    years = calculate_years(df, begin_date, end_date)
    
    # Calculate the Composite Price summary (beginning, ending, and ratio)
    begin_comp, end_comp, ratio_comp = composite_price_summary(df, begin_date, end_date)

    
    # Calculate the CAGR for each metric
    nominal_cagr_result = nominal_return_cagr(df, begin_date, end_date, years)
    total_return_cagr_result = total_return_cagr(df, begin_date, end_date, years)
    real_return_no_div_cagr_result = real_return_no_div_cagr(df, begin_date, end_date, years)
    real_return_with_div_cagr_result = real_return_with_div_cagr(df, begin_date, end_date, years)
    begin_cpi, end_cpi, inflation_cagr_result = inflation_cagr(df, begin_date, end_date, years)

    # Calculate the nominal ending value using Total Return (with Dividend Reinvested) CAGR
    nominal_ending_value = calculate_ending_value(initial_investment, total_return_cagr_result, years)

    # Calculate the inflation-adjusted ending value using Real Total Return (with Dividend Reinvested) CAGR
    inflation_adjusted_ending_value = calculate_ending_value(initial_investment, real_return_with_div_cagr_result, years)

    st.subheader("**Nominal Values**")
    # First row of KPIs (Nominal returns side by side)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Annual Return (Without Dividends)", value=f"{nominal_cagr_result}%")
    with col2:
        st.metric(label="Annual Return (With Dividends)", value=f"{total_return_cagr_result}%")
    with col3:
        st.metric(label=f"${initial_investment:,.0f} turns into", value=f"${nominal_ending_value:,.0f}")

    st.subheader("**Adjusted for Inflation**")
    # Second row of KPIs (Inflation-Adjusted returns side by side)
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric(label="Annual Return (Without Dividends)", value=f"{real_return_no_div_cagr_result}%")
    with col5:
        st.metric(label="Annual Return (With Dividends)", value=f"{real_return_with_div_cagr_result}%")
    with col6:
        st.metric(label=f"${initial_investment:,.0f} turns into", value=f"${inflation_adjusted_ending_value:,.0f}")

    st.subheader(" Values")
    # Third row of KPIs (CPI side by side)
    col7, col8, col9 = st.columns(3)
    with col7:
        st.metric(label="Beginning CPI Value", value=f"{begin_cpi:,.2f}")
    with col8:
        st.metric(label="Ending CPI Value", value=f"{end_cpi:,.2f}")
    with col9:
        st.metric(label="CPI CAGR", value=f"{inflation_cagr_result}%")

    st.subheader("Dividends")
    # Fourth row of KPIs (Dividends side by side)
    col10, col11, col12 = st.columns(3)
    begin_dividend, end_dividend, div_cagr = dividend_cagr(df, begin_date, end_date, years)
    with col10:
        st.metric(label="Beginning Dividends", value=f"${begin_dividend:,.2f}")
    with col11:
        st.metric(label="Ending Dividends", value=f"${end_dividend:,.2f}")
    with col12:
        st.metric(label="Dividend CAGR", value=f"{div_cagr}%")

    # Composite Price Summary Section
    st.subheader("Composite Price Summary")
    col13, col14, col15 = st.columns(3)
    with col13:
        st.metric(label="Beginning Composite Price", value=f"{begin_comp:,.2f}")
    with col14:
        st.metric(label="Ending Composite Price", value=f"{end_comp:,.2f}")
    with col15:
        label = "Increased by a factor of" if ratio_comp >= 1 else "Decreased by a factor of"
        st.metric(label=label, value=f"{ratio_comp:.2f}")

    # Dividend Summary
    st.subheader("Dividend Summary")
    col16, col17, col18 = st.columns(3)
    with col16:
        st.metric(label="Beginning Dividends", value=f"{begin_dividend:,.2f}")
    with col17:
        st.metric(label="Ending Dividends", value=f"{end_dividend:,.2f}")
    dividend_ratio = end_dividend / begin_dividend
    with col18:
        label = "Increased by a factor of" if dividend_ratio >= 1 else "Decreased by a factor of"
        st.metric(label=label, value=f"{dividend_ratio:.2f}")

    # Inflation Summary
    st.subheader("Inflation Summary")
    col19, col20, col21 = st.columns(3)
    with col19:
        st.metric(label="Beginning CPI Index", value=f"{begin_cpi:,.2f}")
    with col20:
        st.metric(label="Ending CPI Index", value=f"{end_cpi:,.2f}")
    cpi_ratio = end_cpi / begin_cpi
    with col21:
        label = "Increased by a factor of" if cpi_ratio >= 1 else "Decreased by a factor of"
        st.metric(label=label, value=f"{cpi_ratio:.2f}")        
        



    # Text Summary
    st.subheader("Summary")
    st.write(
        f"Over this period, the compounded annual growth rate for the SP500 Index, with dividends reinvested was {total_return_cagr_result}%, "
        f"while the inflation-adjusted compounded annual growth rate, with dividends reinvested was {real_return_with_div_cagr_result}%. "
        f"The compounded annual growth rate for the CPI during this same time was {inflation_cagr_result}%. "

    )

    st.write(f"The SP500 increased by a factor of {ratio_comp:.2f}, Dividends increased by a factor of {dividend_ratio:.2f}, while inflation increased by a factor of {cpi_ratio:.2f}."
    )        

