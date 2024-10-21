import pandas as pd

# Load your Excel data into a DataFrame
df = pd.read_excel('data.xlsx')

# Function to calculate Nominal Return No Dividends
def nominal_return_no_dividends(df, begin_date, end_date):
    begin_price = df.loc[df['Date'] == begin_date, 'Composite Price'].values[0]
    end_price = df.loc[df['Date'] == end_date, 'Composite Price'].values[0]
    nominal_return = ((end_price / begin_price) - 1) * 100
    return round(nominal_return, 2)

# Function to calculate Total Return with Dividend Reinvested
def total_return_with_dividends(df, begin_date, end_date):
    begin_total_return_price = df.loc[df['Date'] == begin_date, 'Nominal Total Return Price'].values[0]
    end_total_return_price = df.loc[df['Date'] == end_date, 'Nominal Total Return Price'].values[0]
    total_return = ((end_total_return_price / begin_total_return_price) - 1) * 100
    return round(total_return, 2)

# Function to calculate Inflation Rate
def inflation_rate(df, begin_date, end_date):
    begin_cpi = df.loc[df['Date'] == begin_date, 'CPI'].values[0]
    end_cpi = df.loc[df['Date'] == end_date, 'CPI'].values[0]
    inflation = ((end_cpi / begin_cpi) - 1) * 100
    return round(inflation, 2)

# Function to calculate Real Total Return No Dividend Reinvestment
def real_total_return_no_dividends(df, begin_date, end_date):
    begin_real_price = df.loc[df['Date'] == begin_date, 'Real Price'].values[0]
    end_real_price = df.loc[df['Date'] == end_date, 'Real Price'].values[0]
    real_return_no_div = ((end_real_price / begin_real_price) - 1) * 100
    return round(real_return_no_div, 2)

# Function to calculate Real Total Return With Dividend Reinvestment
def real_total_return_with_dividends(df, begin_date, end_date):
    begin_real_total_return_price = df.loc[df['Date'] == begin_date, 'Real Total Return Price'].values[0]
    end_real_total_return_price = df.loc[df['Date'] == end_date, 'Real Total Return Price'].values[0]
    real_return_with_div = ((end_real_total_return_price / begin_real_total_return_price) - 1) * 100
    return round(real_return_with_div, 2)

# Input your beginning and ending dates
begin_date = 1871.01  # Example starting date
end_date = 1872.01    # Example ending date

# Calculate and display the results
nominal_result = nominal_return_no_dividends(df, begin_date, end_date)
total_return_result = total_return_with_dividends(df, begin_date, end_date)
inflation_result = inflation_rate(df, begin_date, end_date)
real_return_no_div_result = real_total_return_no_dividends(df, begin_date, end_date)
real_return_with_div_result = real_total_return_with_dividends(df, begin_date, end_date)

print(f"Nominal Return (No Dividends Reinvested) from {begin_date} to {end_date}: {nominal_result}%")
print(f"Nominal Total Return (With Dividend Reinvested) from {begin_date} to {end_date}: {total_return_result}%")
print(f"Inflation Rate from {begin_date} to {end_date}: {inflation_result}%")
print(f"Real Return (No Dividends Reinvested) from {begin_date} to {end_date}: {real_return_no_div_result}%")
print(f"Real Total Return (With Dividend Reinvested) from {begin_date} to {end_date}: {real_return_with_div_result}%")