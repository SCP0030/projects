

# -*- coding: utf-8 -*-


import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
pd.set_option('display.max_columns', 20)

#%matplotlib inline

#instructions from https://towardsdatascience.com/web-scraping-nba-stats-4b4f8c525994
# NBA season
year = 2019
# URL page we will scraping
url = "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html".format(year)
html = urlopen(url)
soup = BeautifulSoup(html)

# =============================================================================
# The next step is to organize the column headers. 
# We want to extract the text content of each column header and store them into a list.
#  We do this by inspecting the HTML (right-clicking the page and selecting “Inspect Element”) 
#  we see that the 2nd table row is the one that contains the column headers we want. 
#  By looking at the table, we can see the specific HTML tags that we will be using to extract the data:
# =============================================================================

# use findALL() to get the column headers
soup.findAll('tr', limit=2)
# use getText()to extract the text we need into a list
headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
# exclude the first column as we will not need the ranking order from Basketball Reference for the analysis
headers = headers[1:]
#print(headers)

# avoid the first header row
rows = soup.findAll('tr')[1:]
player_stats = [[td.getText() for td in rows[i].findAll('td')]
            for i in range(len(rows))]

stats = pd.DataFrame(player_stats, columns = headers)
stats.index =  [stats['Player']]
stats = stats.replace(r'^\s*$', np.nan, regex=True) #fills in blank spaces with NaN
stats['MP'].fillna(0,inplace=True)
stats['TS%'].fillna(0, inplace=True)

plot_stats = stats[['MP', 'TS%']].copy()
print(stats[stats.Player=='Tyler Ulis'])

stats[stats.Player=='Tyler Ulis'].describe()
stats['MP'] = stats['MP'].astype(float)
stats['TS%'] = stats['TS%'].astype(float)
stats['TrueShooting'] = stats['TS%']
 
 
#print(stats.loc[new_pts.idxmax()])
#print(stats.loc[new_age.idxmax()])


#Index_label = stats[new_pts>30].index  
  
# Print all the labels 
#for i in Index_label:
    #val = ''.join(i)
    #print(val)
    
#sklearn.cluster.KMeans
#sklearn.preprocessing.MinMaxScaler

plot_stats = stats[['MP', 'TrueShooting']].copy()
plot_stats.fillna(1, inplace=True) 
plot_stats = plot_stats.reset_index()
plot_stats2 = plot_stats[(plot_stats.MP  > 500) & (plot_stats.TrueShooting <1)]
#this selects for more than 6 minutes per game 

scaler = MinMaxScaler()
scaler.fit(plot_stats2[['MP']])
#scaler.fit(plot_stats[['TS%']])
plot_stats2['MP'] = scaler.transform(plot_stats2[['MP']])
 

km = KMeans(n_clusters = 4)
y_predicted = km.fit_predict(plot_stats2[['MP','TrueShooting']])
plot_stats2['Cluster'] = y_predicted
print(plot_stats2.head(10))

df1 = plot_stats2[plot_stats2.Cluster == 0]
df2 = plot_stats2[plot_stats2.Cluster == 1]
df3 = plot_stats2[plot_stats2.Cluster == 2]
df4 = plot_stats2[plot_stats2.Cluster == 3]

plt.scatter(df1['MP'], df1['TrueShooting'], color = 'green')
plt.scatter(df2['MP'], df2['TrueShooting'], color = 'red')
plt.scatter(df3['MP'], df3['TrueShooting'], color = 'blue')
plt.scatter(df4['MP'], df4['TrueShooting'], color = 'purple')
plt.xlabel("Minutes played")
plt.ylabel("True Shooting %")
 
plt.show()
 
# =============================================================================
# k_rng = range(1,15)
# sse = []
# for k in k_rng:
#     km = KMeans(n_clusters = k)
#     km.fit(plot_stats2[['MP','TrueShooting']])
#     sse.append(km.inertia_)
# 
# plt.xlabel('K')
# plt.ylabel('Sum of Squared error')
# plt.plot(k_rng, sse)
# plt.show()
# =============================================================================
#elbow method shows either 3 or 4