# coding: utf-8

import pandas as pd
import numpy as np
import csv
import sys
import re
from tqdm import tqdm
import graphlab as gl
import graphlab.aggregate as agg

#Importing Twitter Text Data

tweetTrack = open("tweet_text_26.csv", "r")
tweet_text = csv.reader(tweetTrack)

data=[]
for row in tweet_text:
    data.append(row)

# Checking The rows with number of columns more than 7
index=[]
for k in range(len(data)):
    if len(data[k]) > 7:
        index.append(k)

# Managing the rows with number of columns more than 7 according to the general format

for k in range(len(index)):
    x= data[index[k]]
    data[index[k]] = x[:2] + x[len(data[index[k]])-4:]
    x= x[2:len(x)-4]
    y=''.join(x)
    data[index[k]].insert( 2, y)


# Cleaning up the Hyperlinks, Symbols & numbers

for k in range(len(data)):
    data[k][2] = re.sub(r"http\S+", "", data[k][2])
    data[k][2]=re.sub(r'[^\w]', ' ', data[k][2])
    data[k][2] = ''.join([i for i in data[k][2] if not i.isdigit()])


# Moving data into a pandas dataframe

df_text = pd.DataFrame(data, columns = ["tweet_id", "user_id","text","lat","long","a","b"])

# Dropping unnecessary columns

df_text.drop(df_text.columns[[0,3,4,5,6]], axis=1, inplace=True)

# Dumping Data into csv

df_text.to_csv("df_text.csv")

# Dropping dataframe from memory

df_text.drop(df_text.index, inplace=True)

# Importing saved csv file in graphlab sframe

sf_text = gl.SFrame.read_csv("df_text.csv")

# Obtaining the tweet count for each user

user_count = sf_text.groupby(key_columns='user_id',operations={'count': agg.COUNT()})


# Selecting users with more than 10 tweets for processing

user_count = user_count[user_count['count'] > 10]

sf_text.remove_column('X1')

# Obtaining the tweets for only selected users
final_sf=sf_text.join(user_count,on='user_id', how='inner')

final_sf.remove_column('count')

# saving data for processing in spark

#final_sf.save('tweet_sf.json', format='json')