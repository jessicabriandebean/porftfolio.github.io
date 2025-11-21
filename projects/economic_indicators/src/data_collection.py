import pandas as pd
from fredapi import Fred
fred = Fred(api_key='e1ab1d32d6233f0589e41d8a74f37174')
# Collect data
unemployment = fred.get_series('UNRATE')
cpi = fred.get_series('CPIAUCSL')
gdp = fred.get_series('GDP')
dff = fred.get_series('DFF')
consumer = fred.get_series('UMCSENT')


# Combine into dataframe
economic_data = pd.DataFrame({
'unemployment': unemployment,
'cpi': cpi,
'gdp': gdp,
'dff': dff,
'consumer' : consumer
})

# Save to CSV
economic_data.to_csv("economic_data.csv", index=False)