'''This is a Submission for DTI coding challenge 2020 Part 1
'''
import pandas as pd
import numpy as np
import timeit
import os
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
os.chdir(r'C:\Users\Dell\Desktop\Capstone\Temp')
News = pd.read_csv('all-the-news-2-1.csv',low_memory=False)
def display_all(df):
    with pd.option_context('display.max_rows', 1000):
        with pd.option_context('display.max_columns', 1000):
            display(df)

# Cleaning News data
display_all(News.head().transpose())
News.info()
News.date = pd.to_datetime(News.date)
News.drop(columns=['year', 'month', 'day', 'url', 'section' ], inplace=True)
News['year'] = pd.DatetimeIndex(News.date).year
News.year.isnull().sum()
News.dropna(subset=['year'], inplace=True)
News.title.isnull().sum()
News.dropna(subset=['title'], inplace=True)

# Extracting data from 2019 and 2020
News_19_20 = News[(News.year == 2019) (News.year == 2020)]

# Sentiment analysis. I didn't do any lemmatizing, stopwords or punctuation removing,
# as the common libaries don't work well with our project. I don't have time to find a proper one.
sid = SentimentIntensityAnalyzer()
News_19_20['title_sentiment'] = News_19_20.apply(lambda row: sid.polarity_scores(row.title)['compound'], axis=1)
News_19_20.to_csv('News_19_20_cleaned_senti.csv')
News_19_20_noneutral = News_19_20[(News_19_20.title_sentiment != 0) & (News_19_20.title_sentiment != 0)]
News_19_20_noneutral['article_sentiment_dummy'] = np.where(News_19_20_noneutral.article_sentiment > 0, 1, 0)
News_19_20_noneutral['title_sentiment_dummy'] = np.where(News_19_20_noneutral.title_sentiment > 0, 1, 0)
News_19_20_noneutral.to_csv('News_19_20_noneutral.csv')

# Clearning currency data
fx_dict = {}
for i in range(1, 16):
    if i < 10:
        fx_dict[i] = pd.read_csv('C:\\Users\\Dell\\Desktop\\Capstone\\Dataset_fx\\Ask\\DAT_NT_GBPUSD_T_ASK_20190{}.csv'.format(i), header=None)
    elif i < 13: 
        fx_dict[i] = pd.read_csv('C:\\Users\\Dell\\Desktop\\Capstone\\Dataset_fx\\Ask\\DAT_NT_GBPUSD_T_ASK_2019{}.csv'.format(i), header=None)
    else: 
        fx_dict[i] = pd.read_csv('C:\\Users\\Dell\\Desktop\\Capstone\\Dataset_fx\\Ask\\DAT_NT_GBPUSD_T_ASK_20200{}.csv'.format(i-12), header=None)
    if i > 1:
        fx_all = fx_all.append(fx_dict[i])
    else:
        fx_all = fx_dict[i]
fx_all = fx_all[0].str.split(pat=';', expand=True)        
fx_all[0] = pd.to_datetime(fx_all[0], format='%Y%m%d %H%M%S')
fx_all.drop(columns = 2, inplace=True)
fx_all.columns = ['time', 'ask_quote']
fx_all.shape
fx_all.to_csv('fx_19_20.cleaned.csv')
fx_all_201902 = fx_all[(fx_all.time.dt.year == 2019) & (fx_all.time.dt.month == 2)]
News_19_20_noneutral_201902 = News_19_20_noneutral[(News_19_20_noneutral.date.dt.year == 2019) & (News_19_20_noneutral.date.dt.month == 2)]

# Drawing a joint time series plot using one month data

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('GBP/USD ask quote price', color=color)
ax1.plot(fx_all_201902.time, fx_all_201902.ask_quote, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx() 

color = 'tab:blue'
ax2.set_ylabel('Sentiment score', color=color)
ax2.plot(News_19_20_noneutral_201902.date, News_19_20_noneutral_201902.article_sentiment_dummy, color=color, marker='.')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
'''This is a Submission for DTI coding challenge 2020 Part 2
'''

# Import packages, read files and simple EDA
import pandas as pd
import numpy as np
import re
import os
import timeit
def display_all(df):
    with pd.option_context('display.max_rows', 1000):
        with pd.option_context('display.max_columns', 1000):
            display(df)
data_17 = pd.read_csv('PartD_Prescriber_PUF_NPI_17.txt', sep='\t')
data_16 = pd.read_csv('PartD_Prescriber_PUF_NPI_16.txt', sep='\t')
display_all(data_17.tail(10).transpose().sort_index())

# Answer to the quesion: In 2017, what was the average number of beneficiaries per provider...
data_17[data_17.bene_count > 10].bene_count.sum() / data_17.shape[0]

# Answer to the quesion: Work out for each Specialty the fraction of drug claims that are for brand-name drugs...
by_specialty = data_17[data_17.total_claim_count >= 1000].groupby('specialty_description')
(by_specialty.brand_claim_count.count() / by_specialty.total_claim_count.count()).std()

# Answer to the quesion: Let's find which states have surprisingly high supply of opioids, conditioned on specialty....
data_17.loc[:, 'len_opioid_avg'] = data_17.opioid_day_supply / data_17.opioid_claim_count
data_17_noless100 = data_17.groupby(['nppes_provider_state', 'specialty_description']).filter(lambda group: group.npi.count() >= 100)
mean_spec_state = data_17_noless100.groupby(['specialty_description', 'nppes_provider_state']).len_opioid_avg.mean()
mean_spec = data_17_noless100.groupby('specialty_description').len_opioid_avg.mean()
max(mean_spec_state / mean_spec)

# Answer to the quesion: For each provider, estimate the length of the average prescription from the total_day...
(data_17.total_day_supply / data_17.total_claim_count).median()

# Answer to the quesion: Find the ratio of beneficiaries with opioid prescriptions to beneficiaries with antibiotics...
by_state = data_17.groupby('nppes_provider_state')
Q4 = by_state.opioid_claim_count.count() / by_state.antibiotic_claim_count.count()
Q4.max() - Q4.min()

# Answer to the quesion: For each provider where the relevant columns are not suppressed, work out the fraction of claims for beneficiaries age 65...
Q5_65 = data_17.total_claim_count_ge65 / data_17.total_claim_count
Q5_lowinc = data_17.lis_claim_count / data_17.total_claim_count
Q5_65.corr(Q5_lowinc)

# Answer to the quesion: For each provider for whom the information is not suppressed, figure out the average...
data_17.loc[:, 'cost_per_day'] = data_17.total_drug_cost / data_17.total_day_supply
data_16.loc[:, 'cost_per_day'] = data_16.total_drug_cost / data_16.total_day_supply
inflation = data_17.cost_per_day / data_16.cost_per_day - 1

## Drop the inf values and NaN values
for i in range(len(inflation)):
    if inflation[i] == np.inf:
        print(True)
        inflation.drop(index=i, inplace=True)
inflation.dropna(inplace=True)
inflation.mean()

# Answer to the quesion: Consider all providers with a defined specialty in both years. Find the fraction of providers...
merged = pd.merge(data_17[['npi', 'specialty_description']], data_16[['npi', 'specialty_description']], on='npi', how='outer')
merged.columns = ['npi', 'specialty_description_17', 'specialty_description_16']
display_all(merged)

## Eyeballing the name changes
display_all(print(pd.Series(merged.specialty_description_17.unique())[pd.Series(merged.specialty_description_17.unique()).notnull()].sort_values().to_string(index = False)))
display_all(print(pd.Series(merged.specialty_description_16.unique())[pd.Series(merged.specialty_description_16.unique()).notnull()].sort_values().to_string(index = False)))

size_spec_17 = pd.DataFrame(merged.groupby('specialty_description_17').size())
size_spec_17.reset_index(inplace=True)
size_spec_16_1000 = pd.DataFrame(merged.groupby('specialty_description_16').size()[merged.groupby('specialty_description_16').size()>=1000])
size_spec_16_1000.reset_index(inplace=True)
spec_16_17 = pd.merge(size_spec_17, size_spec_16_1000, left_on='specialty_description_17', right_on='specialty_description_16', how='right')
display_all(spec_16_17)

## Drop the rows with changed specialty names
spec_16_17.dropna(inplace=True)
(spec_16_17['0_y'] / spec_16_17['0_x'] - 1).max()
'''This is a Submission for DTI coding challenge 2020 Part 3
'''
# Import packages
import random
import numpy

# Define function
## This function uses random.sample to draw all samples without replacement and put them into a list
## then we calculate payment each time using corresponding elements in the list.
import random
import numpy
def payment_bootstrap(n, m):
    payment_list = []
    for k in range(m):
        sample_list = random.sample(range(1, n+1), n)
        payment = 0
        for i, j in enumerate(sample_list):
            if i == 0:
                payment += j
            else:
                payment += np.absolute(j - sample_list[i-1])
        payment_list.append(payment)
    return payment_list


# Generating outputs
result_10 = payment_bootstrap(10, 1000000)
np.mean(result_10)
np.std(result_10)
result_20 = payment_bootstrap(20, 1000000)
np.mean(result_20)
np.std(result_20)
len([i for i in result_10 if i >= 45]) / 1000000
len([i for i in result_20 if i >= 160]) / 1000000