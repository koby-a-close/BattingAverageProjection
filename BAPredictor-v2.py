# BAPredictor-v2.py
# Created: 09/21/2019

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Load packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


# Import Data
data_dir = '/Users/Koby/PycharmProjects/PhilliesBAModel/Input/'
df_data = pd.read_csv(data_dir + 'batting.csv')

# Get features for initial model training and batting averages from first month of season
X = df_data[['MarApr_BB%', 'MarApr_K%', 'MarApr_ISO', 'MarApr_BABIP', 'MarApr_OBP', 'MarApr_LD%', 'MarApr_GB%',
             'MarApr_FB%', 'MarApr_HR/FB', 'MarApr_O-Swing%', 'MarApr_Z-Swing%', 'MarApr_Swing%', 'MarApr_O-Contact%',
             'MarApr_Z-Contact%', 'MarApr_Contact%']].copy()
y = df_data['MarApr_AVG'].copy()

# Trains and initial model using most of the features given in the dataset. This will be used to determine which
# features to include in the final model. Basic starting parameters were used for the model.
MLR_model1 = LinearRegression()
clf_original = MLR_model1.fit(X,y)
# Evaluate using cross validation and MAE scoring.
CV_score = cross_val_score(MLR_model1, X, y, scoring='neg_mean_absolute_error')
scr = np.mean(CV_score)
print("MAE for All Features: ", scr)
# Print feature importances to ID top features for final model.
print(pd.DataFrame({'Variable':X.columns,
              'Importance': MLR_model1.coef_}).sort_values('Importance', ascending=False))

# # Filter out top features from XGBoost model.
# X_top5 = X[['MarApr_BABIP', 'MarApr_K%', 'MarApr_ISO', 'MarApr_OBP', 'MarApr_BB%']].copy()
# X_top5 = X_top5.rename(columns={'MarApr_ISO':'ISO', 'MarApr_BABIP':'BABIP',
#                                 'MarApr_K%': 'K%', 'MarApr_OBP':'OBP'})
# MLR_BAModel = ElasticNet()
# clf = MLR_BAModel.fit(X_top5, y)
#
# CV_score = cross_val_score(MLR_BAModel, X_top5, y, scoring='neg_mean_absolute_error')
# scr = np.mean(CV_score)
# print("MAE for Top 4 Features: ", scr)
# print(pd.DataFrame({'Variable':X_top5.columns,
#               'Importance': abs(MLR_BAModel.coef_)}).sort_values('Importance', ascending=False))

# Uses MarApr stats and league averages to regress final stats to the mean. Weighted for PA in MarApr to hold trends
# of "regulars" more than players with low playing time.
df_regress = X.copy()
df_regress = df_regress.rename(columns={'MarApr_ISO':'ISO', 'MarApr_BABIP':'BABIP',
                                        'MarApr_K%': 'K%', 'MarApr_OBP':'OBP', 'MarApr_LD%':'LD%', 'MarApr_GB%':'GB%',
                                        'MarApr_FB%':'FB%', 'MarApr_BB%':'BB%', 'MarApr_Swing%':'Swing%',
                                        'MarApr_HR/FB':'HR/FB', 'MarApr_Contact%':'Contact%',
                                        'MarApr_O-Contact%':'O-Contact%', 'MarApr_Z-Contact%':'Z-Contact%',
                                        'MarApr_O-Swing%':'O-Swing%', 'MarApr_Z-Swing%':'Z-Swing%'})
#X_top5.copy()
df_regress['MarApr_PA'] = df_data['MarApr_PA'].copy()
df_regress['Multiplier'] = df_regress['MarApr_PA']/180 # Original value used was 500
df_regress['LD%'] = df_regress['LD%']*df_regress['Multiplier'] + 0.210*(1-df_regress['Multiplier'])
df_regress['GB%'] = df_regress['GB%']*df_regress['Multiplier'] + 0.440*(1-df_regress['Multiplier'])
df_regress['FB%'] = df_regress['FB%']*df_regress['Multiplier'] + 0.350*(1-df_regress['Multiplier'])
df_regress['BABIP'] = df_regress['BABIP']*df_regress['Multiplier'] + 0.300*(1-df_regress['Multiplier'])
df_regress['K%'] = df_regress['K%']*df_regress['Multiplier'] + 0.200*(1-df_regress['Multiplier'])
df_regress['OBP'] = df_regress['OBP']*df_regress['Multiplier'] + 0.320*(1-df_regress['Multiplier'])
df_regress['BB%'] = df_regress['BB%']*df_regress['Multiplier'] + 0.080*(1-df_regress['Multiplier'])
df_regress['ISO'] = df_regress['ISO']*df_regress['Multiplier'] + 0.140*(1-df_regress['Multiplier'])
df_regress['Swing%'] = df_regress['Swing%']*df_regress['Multiplier'] + 0.460*(1-df_regress['Multiplier'])
df_regress['HR/FB'] = df_regress['HR/FB']*df_regress['Multiplier'] + 0.100*(1-df_regress['Multiplier'])
df_regress['Contact%'] = df_regress['Contact%']*df_regress['Multiplier'] + 0.800*(1-df_regress['Multiplier'])
df_regress['O-Contact%'] = df_regress['O-Contact%']*df_regress['Multiplier'] + 0.660*(1-df_regress['Multiplier'])
df_regress['Z-Contact%'] = df_regress['Z-Contact%']*df_regress['Multiplier'] + 0.870*(1-df_regress['Multiplier'])
df_regress['O-Swing%'] = df_regress['O-Swing%']*df_regress['Multiplier'] + 0.300*(1-df_regress['Multiplier'])
df_regress['Z-Swing%'] = df_regress['Z-Swing%']*df_regress['Multiplier'] + 0.650*(1-df_regress['Multiplier'])

# Copies features of interest from df_regressed
X_regressed = df_regress[['BB%', 'K%', 'ISO', 'BABIP', 'OBP', 'LD%', 'GB%', 'FB%', 'HR/FB',
                          'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%']]

print(X.head(10))
print(X_regressed.head(10))

# Makes EoY BA predictions
Final_predictions = clf_original.predict(X_regressed)
sns.distplot(Final_predictions)
max_BA = max(Final_predictions)
min_BA = min(Final_predictions)
print("Max predicted BA: ", max_BA)
print("Min predicted BA: ", min_BA)

# Evaluate the model using actual EoY averages provided
MAE_score = round(mean_absolute_error(df_data['FullSeason_AVG'], Final_predictions), 3)
print("MAE for Full Season BA: ", MAE_score)

MSE_score = round(mean_squared_error(df_data['FullSeason_AVG'], Final_predictions), 3)
print("TSE for Full Season BA: ", MSE_score*309)

plt.figure(2)
plt.plot(df_data['FullSeason_AVG'], Final_predictions, 'bo')
plt.xlabel('Actual BA')
plt.ylabel('Predicted BA')
plt.axis([0.150, 0.375, 0.150, 0.375])
plt.show()

pred = Final_predictions
actual = df_data.FullSeason_AVG.copy()
diff = actual - pred

overestimated = len([i for i in diff if i < 0])
underestimated = len([i for i in diff if i > 0])
ratio = round(overestimated/309*100, 2)
print(overestimated, "batting averages were overestimated by the model (", ratio, "%)")
print(underestimated, "batting averages were underestimated by the model (", 100-ratio, "%)")