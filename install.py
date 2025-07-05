import requests
import zipfile
import os
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

url = 'https://api.worldbank.org/v2/zh/indicator/NY.GDP.PCAP.CD?downloadformat=csv'
response = requests.get(url)
with open('gdp_data.zip', 'wb') as f:
    f.write(response.content)


with zipfile.ZipFile('gdp_data.zip', 'r') as zip_ref:
    zip_ref.extractall('gdp_data')


csv_file = None
for file in os.listdir('gdp_data'):
    if file.endswith('.csv') and 'API_NY.GDP.PCAP.CD_DS2_zh_csv_v2_19649' in file:
        csv_file = os.path.join('gdp_data', file)
        break


df = pd.read_csv(csv_file, skiprows=3)


china = df[df['Country Name'] == '中国'].iloc[0]
years = [str(y) for y in range(1960, 2023)]
china_gdp = china[years].astype(float).dropna()
china_years = pd.to_numeric(china_gdp.index)
china_values = china_gdp.values

plt.figure(figsize=(12, 6))
plt.plot(china_years, china_values, 'r-', label='中国')
plt.title('我国人均GDP数据')
plt.xlabel('年')
plt.ylabel('人均GDP')
plt.grid(True)
plt.legend()
plt.show()


def cal(country_data):
    data = country_data[years].astype(float).dropna()
    data_series = data.iloc[0]
    growth = (data_series / data_series.shift(1) - 1) * 100
    return growth.dropna()


countries = ['中国', '美国', '日本', '德国', '印度']

growth_data = {}
for name in countries:
    country = df[df['Country Name'] == name]
    if not country.empty:
        growth = cal(country)
        growth_data[name] = growth

plt.figure(figsize=(14, 7))
for country, growth in growth_data.items():
    plt.plot(growth.index.astype(int), growth.values,
             label=country, marker='o', markersize=4)

plt.title('我国与其他若干国家的人均GDP增长率')
plt.xlabel('年')
plt.ylabel('增长率')
plt.grid(True)
plt.legend()
plt.show()
