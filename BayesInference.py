
# Load packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.utils import shuffle

# Import Data
data_dir = '/Users/Koby/PycharmProjects/PhilliesBAModel/Input/'
df_data = pd.read_csv(data_dir + 'batting.csv')

# Visualize distribution of current batting averages
sns.distplot(df_data['MarApr_AVG'])

# Calculate mean and variance or current batting averages
BA_average = df_data['MarApr_AVG'].mean()
BA_var = df_data['MarApr_AVG'].var()

# Uses Method of Moments Estimator to calculate alpha and beta
alpha = BA_average*(((BA_average*(1-BA_average))/BA_var)-1)
beta = (1-BA_average)*(((BA_average*(1-BA_average))/BA_var)-1)

df_predictions = pd.DataFrame()
df_predictions['Name'] = df_data.Name.copy()
df_predictions['At_Bats'] = df_data.MarApr_AB.copy()
df_predictions.At_Bats = df_predictions.At_Bats + alpha + beta
df_predictions['Hits'] = df_data.MarApr_H.copy()
df_predictions.Hits = df_predictions.Hits + alpha

df_predictions['Average'] = df_predictions.Hits/df_predictions.At_Bats

plt.plot(df_predictions.Average, df_data.FullSeason_AVG, 'ro')
plt.show()

r_squared = r2_score(df_data.FullSeason_AVG, df_predictions.Average)
print(r_squared)
x=1


x=1