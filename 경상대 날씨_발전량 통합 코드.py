import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df1 = pd.read_csv('C:\\dd\\경상대학교 일별 날씨1.csv')
df2020 = pd.read_csv('C:\\dd\\경상대학교 시간별 발전량2020.csv')
df2019 = pd.read_csv('C:\\dd\\경상대학교 시간별 발전량2019.csv')
df2018 = pd.read_csv('C:\\dd\\경상대학교 시간별 발전량2018.csv')
df2017 = pd.read_csv('C:\\dd\\경상대학교 시간별 발전량2017.csv')


df2017['Total'] = df2017.iloc[:, 3:].sum(axis=1)
df2018['Total'] = df2018.iloc[:, 3:].sum(axis=1)
df2019['Total'] = df2019.iloc[:, 3:].sum(axis=1)
df2020['Total'] = df2020.iloc[:, 3:].sum(axis=1)

df17 = df2017.iloc[:, [0,-1]]
df17.columns = ['day', 'total']
df18 = df2018.iloc[:, [0,-1]]
df18.columns = ['day', 'total']
df19 = df2019.iloc[:, [0,-1]]
df19.columns = ['day', 'total']
df20 = df2020.iloc[:, [0,-1]]
df20.columns = ['day', 'total']

df_new = df17.append([df18, df19, df20], sort=False)

df1.rename(columns={'일시':'day'}, inplace=True)

df1 = pd.merge(df1, df_new, on='day', how='left')

df1.to_csv('C:\\dd\\fin.csv', header=False, index=False, encoding='utf-8')
