In [1]:
#Import our libraries and datasets 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import random
import seaborn as sns 
from fbprophet import Prophet
In [2]:
#upload dataset
from google.colab import files
uploaded = files.upload()
Saving avocado.csv to avocado.csv
In [3]:
#Selecting the dataset
import io
avocado_df = pd.read_csv(io.BytesIO(uploaded['avocado.csv']))
In [4]:
#Executing from your local systems
#avocado_df = pd.read_csv('avocado.csv')
In [5]:
#Description if our dataset 
#It gives the list of attributes in our Dataset

avocado_df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 18249 entries, 0 to 18248
Data columns (total 14 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Unnamed: 0    18249 non-null  int64  
 1   Date          18249 non-null  object 
 2   AveragePrice  18249 non-null  float64
 3   Total Volume  18249 non-null  float64
 4   4046          18249 non-null  float64
 5   4225          18249 non-null  float64
 6   4770          18249 non-null  float64
 7   Total Bags    18249 non-null  float64
 8   Small Bags    18249 non-null  float64
 9   Large Bags    18249 non-null  float64
 10  XLarge Bags   18249 non-null  float64
 11  type          18249 non-null  object 
 12  year          18249 non-null  int64  
 13  region        18249 non-null  object 
dtypes: float64(9), int64(2), object(3)
memory usage: 1.9+ MB
In [6]:
#Last 10 values
avocado_df.tail(10)
Out[6]:
Unnamed: 0	Date	AveragePrice	Total Volume	4046	4225	4770	Total Bags	Small Bags	Large Bags	XLarge Bags	type	year	region
18239	2	2018-03-11	1.56	22128.42	2162.67	3194.25	8.93	16762.57	16510.32	252.25	0.0	organic	2018	WestTexNewMexico
18240	3	2018-03-04	1.54	17393.30	1832.24	1905.57	0.00	13655.49	13401.93	253.56	0.0	organic	2018	WestTexNewMexico
18241	4	2018-02-25	1.57	18421.24	1974.26	2482.65	0.00	13964.33	13698.27	266.06	0.0	organic	2018	WestTexNewMexico
18242	5	2018-02-18	1.56	17597.12	1892.05	1928.36	0.00	13776.71	13553.53	223.18	0.0	organic	2018	WestTexNewMexico
18243	6	2018-02-11	1.57	15986.17	1924.28	1368.32	0.00	12693.57	12437.35	256.22	0.0	organic	2018	WestTexNewMexico
18244	7	2018-02-04	1.63	17074.83	2046.96	1529.20	0.00	13498.67	13066.82	431.85	0.0	organic	2018	WestTexNewMexico
18245	8	2018-01-28	1.71	13888.04	1191.70	3431.50	0.00	9264.84	8940.04	324.80	0.0	organic	2018	WestTexNewMexico
18246	9	2018-01-21	1.87	13766.76	1191.92	2452.79	727.94	9394.11	9351.80	42.31	0.0	organic	2018	WestTexNewMexico
18247	10	2018-01-14	1.93	16205.22	1527.63	2981.04	727.01	10969.54	10919.54	50.00	0.0	organic	2018	WestTexNewMexico
18248	11	2018-01-07	1.62	17489.58	2894.77	2356.13	224.53	12014.15	11988.14	26.01	0.0	organic	2018	WestTexNewMexico
In [7]:
#Mean median and stansard deviation (Statistical Analysis)

avocado_df.describe()
Out[7]:
Unnamed: 0	AveragePrice	Total Volume	4046	4225	4770	Total Bags	Small Bags	Large Bags	XLarge Bags	year
count	18249.000000	18249.000000	1.824900e+04	1.824900e+04	1.824900e+04	1.824900e+04	1.824900e+04	1.824900e+04	1.824900e+04	18249.000000	18249.000000
mean	24.232232	1.405978	8.506440e+05	2.930084e+05	2.951546e+05	2.283974e+04	2.396392e+05	1.821947e+05	5.433809e+04	3106.426507	2016.147899
std	15.481045	0.402677	3.453545e+06	1.264989e+06	1.204120e+06	1.074641e+05	9.862424e+05	7.461785e+05	2.439660e+05	17692.894652	0.939938
min	0.000000	0.440000	8.456000e+01	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000	2015.000000
25%	10.000000	1.100000	1.083858e+04	8.540700e+02	3.008780e+03	0.000000e+00	5.088640e+03	2.849420e+03	1.274700e+02	0.000000	2015.000000
50%	24.000000	1.370000	1.073768e+05	8.645300e+03	2.906102e+04	1.849900e+02	3.974383e+04	2.636282e+04	2.647710e+03	0.000000	2016.000000
75%	38.000000	1.660000	4.329623e+05	1.110202e+05	1.502069e+05	6.243420e+03	1.107834e+05	8.333767e+04	2.202925e+04	132.500000	2017.000000
max	52.000000	3.250000	6.250565e+07	2.274362e+07	2.047057e+07	2.546439e+06	1.937313e+07	1.338459e+07	5.719097e+06	551693.650000	2018.000000
In [7]:

In [8]:
#Data Preprocessing
#Checking for null values

avocado_df.isnull().sum()
Out[8]:
Unnamed: 0      0
Date            0
AveragePrice    0
Total Volume    0
4046            0
4225            0
4770            0
Total Bags      0
Small Bags      0
Large Bags      0
XLarge Bags     0
type            0
year            0
region          0
dtype: int64
In [9]:
#Sort our data according to Data

avocado_df = avocado_df.sort_values('Date')
avocado_df
Out[9]:
Unnamed: 0	Date	AveragePrice	Total Volume	4046	4225	4770	Total Bags	Small Bags	Large Bags	XLarge Bags	type	year	region
11569	51	2015-01-04	1.75	27365.89	9307.34	3844.81	615.28	13598.46	13061.10	537.36	0.00	organic	2015	Southeast
9593	51	2015-01-04	1.49	17723.17	1189.35	15628.27	0.00	905.55	905.55	0.00	0.00	organic	2015	Chicago
10009	51	2015-01-04	1.68	2896.72	161.68	206.96	0.00	2528.08	2528.08	0.00	0.00	organic	2015	HarrisburgScranton
1819	51	2015-01-04	1.52	54956.80	3013.04	35456.88	1561.70	14925.18	11264.80	3660.38	0.00	conventional	2015	Pittsburgh
9333	51	2015-01-04	1.64	1505.12	1.27	1129.50	0.00	374.35	186.67	187.68	0.00	organic	2015	Boise
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
8574	0	2018-03-25	1.36	908202.13	142681.06	463136.28	174975.75	127409.04	103579.41	22467.04	1362.59	conventional	2018	Chicago
9018	0	2018-03-25	0.70	9010588.32	3999735.71	966589.50	30130.82	4014132.29	3398569.92	546409.74	69152.63	conventional	2018	SouthCentral
18141	0	2018-03-25	1.42	163496.70	29253.30	5080.04	0.00	129163.36	109052.26	20111.10	0.00	organic	2018	SouthCentral
17673	0	2018-03-25	1.70	190257.38	29644.09	70982.10	0.00	89631.19	89424.11	207.08	0.00	organic	2018	California
8814	0	2018-03-25	1.34	1774776.77	63905.98	908653.71	843.45	801373.63	774634.09	23833.93	2905.61	conventional	2018	NewYork
18249 rows × 14 columns

In [9]:

In [10]:
#Data Visualization
#plot distribution of Average Price and Date

plt.figure(figsize = (10, 6))
plt.plot(avocado_df['Date'],avocado_df['AveragePrice'])
Out[10]:
[<matplotlib.lines.Line2D at 0x7fc696da1fd0>]

In [11]:
#Plot distribution of Average 

plt.figure(figsize=(10,6))
sns.distplot(avocado_df['AveragePrice'], color = 'g')
/usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning:

`distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).

Out[11]:
<matplotlib.axes._subplots.AxesSubplot at 0x7fc696b7bcd0>

In [12]:
#Bar Graph of region vs AveragePrice

plt.figure(figsize=(20,8))
ax=sns.barplot(x='region',y='AveragePrice',data = avocado_df, palette='rocket')

In [13]:
#violin plot for different types of avocado

sns.violinplot(y='AveragePrice', x='type',data = avocado_df)
Out[13]:
<matplotlib.axes._subplots.AxesSubplot at 0x7fc68b8ca110>

In [14]:
#price of conventional type of avocado vs region

conventional = sns.catplot('AveragePrice','region',data = avocado_df[avocado_df['type']== 'conventional'],hue = 'year',height = 20)
/usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning:

Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.


In [15]:
#price of organic type of avocado vs region

organic = sns.catplot('AveragePrice','region',data = avocado_df[avocado_df['type']== 'organic'],hue = 'year',height = 20)
/usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning:

Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.


In [16]:
avocado_df
Out[16]:
Unnamed: 0	Date	AveragePrice	Total Volume	4046	4225	4770	Total Bags	Small Bags	Large Bags	XLarge Bags	type	year	region
11569	51	2015-01-04	1.75	27365.89	9307.34	3844.81	615.28	13598.46	13061.10	537.36	0.00	organic	2015	Southeast
9593	51	2015-01-04	1.49	17723.17	1189.35	15628.27	0.00	905.55	905.55	0.00	0.00	organic	2015	Chicago
10009	51	2015-01-04	1.68	2896.72	161.68	206.96	0.00	2528.08	2528.08	0.00	0.00	organic	2015	HarrisburgScranton
1819	51	2015-01-04	1.52	54956.80	3013.04	35456.88	1561.70	14925.18	11264.80	3660.38	0.00	conventional	2015	Pittsburgh
9333	51	2015-01-04	1.64	1505.12	1.27	1129.50	0.00	374.35	186.67	187.68	0.00	organic	2015	Boise
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
8574	0	2018-03-25	1.36	908202.13	142681.06	463136.28	174975.75	127409.04	103579.41	22467.04	1362.59	conventional	2018	Chicago
9018	0	2018-03-25	0.70	9010588.32	3999735.71	966589.50	30130.82	4014132.29	3398569.92	546409.74	69152.63	conventional	2018	SouthCentral
18141	0	2018-03-25	1.42	163496.70	29253.30	5080.04	0.00	129163.36	109052.26	20111.10	0.00	organic	2018	SouthCentral
17673	0	2018-03-25	1.70	190257.38	29644.09	70982.10	0.00	89631.19	89424.11	207.08	0.00	organic	2018	California
8814	0	2018-03-25	1.34	1774776.77	63905.98	908653.71	843.45	801373.63	774634.09	23833.93	2905.61	conventional	2018	NewYork
18249 rows × 14 columns

In [17]:
#First  5 values of our data set

avocado_df.head()
Out[17]:
Unnamed: 0	Date	AveragePrice	Total Volume	4046	4225	4770	Total Bags	Small Bags	Large Bags	XLarge Bags	type	year	region
11569	51	2015-01-04	1.75	27365.89	9307.34	3844.81	615.28	13598.46	13061.10	537.36	0.0	organic	2015	Southeast
9593	51	2015-01-04	1.49	17723.17	1189.35	15628.27	0.00	905.55	905.55	0.00	0.0	organic	2015	Chicago
10009	51	2015-01-04	1.68	2896.72	161.68	206.96	0.00	2528.08	2528.08	0.00	0.0	organic	2015	HarrisburgScranton
1819	51	2015-01-04	1.52	54956.80	3013.04	35456.88	1561.70	14925.18	11264.80	3660.38	0.0	conventional	2015	Pittsburgh
9333	51	2015-01-04	1.64	1505.12	1.27	1129.50	0.00	374.35	186.67	187.68	0.0	organic	2015	Boise
In [18]:
#Selecting the required Attributes for Data mining

avocado_prophet_df = avocado_df[['Date', 'AveragePrice']]
avocado_prophet_df
Out[18]:
Date	AveragePrice
11569	2015-01-04	1.75
9593	2015-01-04	1.49
10009	2015-01-04	1.68
1819	2015-01-04	1.52
9333	2015-01-04	1.64
...	...	...
8574	2018-03-25	1.36
9018	2018-03-25	0.70
18141	2018-03-25	1.42
17673	2018-03-25	1.70
8814	2018-03-25	1.34
18249 rows × 2 columns

In [19]:
#renaming the coulmns

avocado_prophet_df.rename(columns={'Date':'ds','AveragePrice' : 'y'},inplace='true')
avocado_prophet_df
/usr/local/lib/python3.7/dist-packages/pandas/core/frame.py:4308: SettingWithCopyWarning:


A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

Out[19]:
ds	y
11569	2015-01-04	1.75
9593	2015-01-04	1.49
10009	2015-01-04	1.68
1819	2015-01-04	1.52
9333	2015-01-04	1.64
...	...	...
8574	2018-03-25	1.36
9018	2018-03-25	0.70
18141	2018-03-25	1.42
17673	2018-03-25	1.70
8814	2018-03-25	1.34
18249 rows × 2 columns

In [20]:
#Fitting a model for training data 

m = Prophet()
m.fit(avocado_prophet_df)
INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
Out[20]:
<fbprophet.forecaster.Prophet at 0x7fc68b7c4510>
In [21]:
#forecasting the future

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
In [22]:
#outcome 
#trend generation
forecast
Out[22]:
ds	trend	yhat_lower	yhat_upper	trend_lower	trend_upper	additive_terms	additive_terms_lower	additive_terms_upper	yearly	yearly_lower	yearly_upper	multiplicative_terms	multiplicative_terms_lower	multiplicative_terms_upper	yhat
0	2015-01-04	1.498818	0.911961	1.896654	1.498818	1.498818	-0.113604	-0.113604	-0.113604	-0.113604	-0.113604	-0.113604	0.0	0.0	0.0	1.385214
1	2015-01-11	1.493637	0.901532	1.843259	1.493637	1.493637	-0.105192	-0.105192	-0.105192	-0.105192	-0.105192	-0.105192	0.0	0.0	0.0	1.388445
2	2015-01-18	1.488455	0.912426	1.877479	1.488455	1.488455	-0.104862	-0.104862	-0.104862	-0.104862	-0.104862	-0.104862	0.0	0.0	0.0	1.383592
3	2015-01-25	1.483273	0.869576	1.830740	1.483273	1.483273	-0.123788	-0.123788	-0.123788	-0.123788	-0.123788	-0.123788	0.0	0.0	0.0	1.359485
4	2015-02-01	1.478091	0.810576	1.808300	1.478091	1.478091	-0.152113	-0.152113	-0.152113	-0.152113	-0.152113	-0.152113	0.0	0.0	0.0	1.325978
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
529	2019-03-21	1.161735	0.537801	1.595805	0.964857	1.346917	-0.086198	-0.086198	-0.086198	-0.086198	-0.086198	-0.086198	0.0	0.0	0.0	1.075537
530	2019-03-22	1.161003	0.557451	1.588914	0.963286	1.346316	-0.084518	-0.084518	-0.084518	-0.084518	-0.084518	-0.084518	0.0	0.0	0.0	1.076485
531	2019-03-23	1.160272	0.574674	1.635688	0.961314	1.346320	-0.082565	-0.082565	-0.082565	-0.082565	-0.082565	-0.082565	0.0	0.0	0.0	1.077707
532	2019-03-24	1.159540	0.539369	1.567668	0.959625	1.346326	-0.080358	-0.080358	-0.080358	-0.080358	-0.080358	-0.080358	0.0	0.0	0.0	1.079182
533	2019-03-25	1.158809	0.560447	1.600469	0.957937	1.346340	-0.077925	-0.077925	-0.077925	-0.077925	-0.077925	-0.077925	0.0	0.0	0.0	1.080884
534 rows × 16 columns

In [23]:
figure = m.plot(forecast, xlabel='Date', ylabel='Price')

In [24]:
#performance during year and month

figure2 = m.plot_components(forecast)

In [25]:
#region specific predictions
#select region  Denver

avocado_df_sample = avocado_df[avocado_df['region']=='Denver']
In [26]:
#Visualization

avocado_df_sample = avocado_df_sample.sort_values('Date')
plt.plot(avocado_df_sample['Date'],avocado_df_sample['AveragePrice'])
INFO:matplotlib.category:Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
INFO:matplotlib.category:Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
Out[26]:
[<matplotlib.lines.Line2D at 0x7fc6879537d0>]

In [27]:
#rename

avocado_df_sample = avocado_df_sample.rename(columns={'Date':'ds','AveragePrice':'y'})
In [28]:
#predictions

m1 = Prophet()
m1.fit (avocado_df_sample)
INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
Out[28]:
<fbprophet.forecaster.Prophet at 0x7fc6883d9e90>
In [29]:
#forecasting the future

future = m1.make_future_dataframe(periods = 365)
forecast1 = m1.predict(future)
In [30]:
#Trends and monthly data 

figure3 = m1.plot(forecast1, xlabel = 'Date' , ylabel = 'Price')
figure4 = m1.plot_components(forecast1)

