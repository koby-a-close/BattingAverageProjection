# Phillies-BAPredictor-v1.py
# Created: 09/21/2019

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Load packages
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns


# Import Data
data_dir = '/Users/Koby/PycharmProjects/PhilliesBAModel/Input/'
df_data = pd.read_csv(data_dir + 'batting.csv')

# Get features for initial model training and batting averages from first month of season
X = df_data[['MarApr_PA', 'MarApr_AB', 'MarApr_BB%', 'MarApr_K%', 'MarApr_ISO', 'MarApr_BABIP', 'MarApr_OBP',
             'MarApr_LD%', 'MarApr_GB%', 'MarApr_FB%', 'MarApr_IFFB%', 'MarApr_HR/FB', 'MarApr_O-Swing%',
             'MarApr_Z-Swing%', 'MarApr_Swing%', 'MarApr_O-Contact%', 'MarApr_Z-Contact%', 'MarApr_Contact%']].copy()
y = df_data['MarApr_AVG'].copy()

# Trains an initial model using most of the features given in the dataset. This will be used to determine which
# features to include in the final model. Basic starting parameters were used for the model.
XGB_model1 = XGBRegressor(learning_rate=0.1, n_estimators=1000, max_depth=3, subsample=0.8)
clf_original = XGB_model1.fit(X,y)
# Evaluate using cross validation and MAE scoring.
CV_score = cross_val_score(XGB_model1, X, y, scoring='neg_mean_absolute_error')
scr = np.mean(CV_score)
print("MAE for All Features: ", scr)
# Print feature importances to ID top features for final model.
print(pd.DataFrame({'Variable':X.columns,
              'Importance':XGB_model1.feature_importances_}).sort_values('Importance', ascending=False))

# Filter out top features. 5 were used.
X_top5 = X[['MarApr_OBP', 'MarApr_BABIP', 'MarApr_K%', 'MarApr_BB%', 'MarApr_ISO']].copy()
X_top5 = X_top5.rename(columns={'MarApr_OBP':'OBP', 'MarApr_BABIP':'BABIP', 'MarApr_K%':'K%', 'MarApr_BB%':'BB%',
                                'MarApr_ISO':'ISO'})
# Model was tuned using some strategies from
# https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e
# and comparing MAE values.
XGB_BAModel = XGBRegressor(learning_rate=0.05, n_estimators=1000, max_depth=5, subsample=0.8, gamma=0)
clf = XGB_BAModel.fit(X_top5, y)
CV_score = cross_val_score(XGB_BAModel, X_top5, y, scoring='neg_mean_absolute_error')
scr = np.mean(CV_score)
print("MAE for Top 5 Features: ", scr)
print(pd.DataFrame({'Variable':X_top5.columns,
              'Importance':XGB_BAModel.feature_importances_}).sort_values('Importance', ascending=False))


# Uses MarApr stats and league averages to regress final stats to the mean. Weighted for PA in MarApr to hold trends
# of "regulars" more than players with low playing time. The 'Multiplier' value was adjusted to get a more realistic
# distribution of batting averages. Using a full 500 expected plate appearance regressed values to the mean too heavily.
df_regress = X_top5.copy()
df_regress['MarApr_PA'] = df_data['MarApr_PA'].copy()
df_regress['Multiplier'] = df_regress['MarApr_PA']/180 # Original value used was 500
df_regress['OBP'] = df_regress['OBP']*df_regress['Multiplier'] + 0.320*(1-df_regress['Multiplier'])
df_regress['BABIP'] = df_regress['BABIP']*df_regress['Multiplier'] + 0.300*(1-df_regress['Multiplier'])
df_regress['K%'] = df_regress['K%']*df_regress['Multiplier'] + 0.200*(1-df_regress['Multiplier'])
df_regress['BB%'] = df_regress['BB%']*df_regress['Multiplier'] + 0.080*(1-df_regress['Multiplier'])
df_regress['ISO'] = df_regress['ISO']*df_regress['Multiplier'] + 0.140*(1-df_regress['Multiplier'])

# Copies features of interest from df_regressed
X_regressed = df_regress[['OBP', 'BABIP', 'K%', 'BB%', 'ISO']]

# Makes EoY BA predictions
Final_predictions = clf.predict(X_regressed)
# Print distribution plot of the predictions as well as max and min values. This was used to adjust the multiplier
# mentioned above.
sns.distplot(Final_predictions)
max_BA = max(Final_predictions)
min_BA = min(Final_predictions)
print("Max predicted BA: ", max_BA)
print("Min predicted BA: ", min_BA)

# Evaluate the model using actual EoY averages provided
score = mean_absolute_error(df_data['FullSeason_AVG'], Final_predictions)
print("MAE for EoY BA: ", score)
# R-squared was also used on the final predicitions as the values would  have a value of 1 if they were perfect.
r_squared = r2_score(df_data.FullSeason_AVG, Final_predictions)
print("R-Squared for EoY BA: ", r_squared)

# Plots the predictions compared to the actual full season batting averages for visualization
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



