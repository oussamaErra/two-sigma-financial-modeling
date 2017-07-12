import kagglegym
import numpy as np
import pandas as pd
from scipy import spatial
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression, LassoCV, RidgeCV, BayesianRidge, OrthogonalMatchingPursuit, HuberRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

# class SwingerRegressor(BaseEstimator, RegressorMixin):
#     def __init__(self, base_estimator=BayesianRidge(), swing_val=2350, lower_bound=0.5, upper_bound=3.5):
#         self.base_estimator = base_estimator
#         self.swing_val = swing_val
#         self.lower_bound = lower_bound
#         self.upper_bound = upper_bound
        
#     def fit(self, X, y):
#         self.base_estimator.fit(X,y)
#         return self
        
#     def predict(self, X):
#         pred = self.base_estimator.predict(X)
#         std = pred.std()
#         multiplier = max(min(std*self.swing_val, self.upper_bound), self.lower_bound)
#         return pred * multiplier



# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

#train = observation.train
col_to_use = ['timestamp','technical_20','technical_30','y']
train = observation.train.copy()
groups = train.groupby('id').size()
selected_group = groups.index[groups==train.timestamp.nunique()]
train = train[train.id.isin(selected_group)]
train = train[col_to_use]

train=train.fillna(0)
train=train.groupby('timestamp').std()

model_std=LinearRegression()
model_std.fit(train[['technical_20','technical_30']],train[['y']] )




train = observation.train
low_y_cut = -0.086093
high_y_cut = 0.093497


# y_is_above_cut = (train.y > high_y_cut)
# y_is_below_cut = (train.y < low_y_cut)
# y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

# train = train.loc[y_is_within_cut,:]
y = train.y
# Note that the first observation we get has a "train" dataframe
# print("Train has {} rows".format(len(observation.train)))
low_y_cut = y.mean() - (1.5 * y.std())
high_y_cut = y.mean() + (1.5 * y.std())

print(low_y_cut)
print(high_y_cut)
# The "target" dataframe is a template for what we need to predict:
# print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))
# top4 = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19', "timestamp"]
top3 = ['technical_30', 'technical_20', "timestamp"]
# top3 = [['technical_30', 'technical_20', 'technical_19', "timestamp"]]
timestamps = np.unique(train.timestamp)

# train.drop(["y", "id"], inplace=True, axis=1)
train = train[top3]
elastics = []


skip = 229
cols = [col for col in train.columns if col != "timestamp"]



normal_weight = 0.5
crazy_weight = 1 - normal_weight
clfs = []
crazy_clfs = []
hubers=[]
indexes = []
means = []

print(normal_weight)
scores = []
for timestamp in timestamps[skip::10]:
    index = (train.timestamp <= timestamp) & (train.timestamp > (timestamp - skip))
    
    period = train[index][cols].copy()
    anti_period = train[~index][cols].copy()
    temp_y = y[index]
    anti_y = y[~index]
    mean_values = period.median(axis=0)
    period.fillna(0, inplace=True)
    anti_period.fillna(0, inplace=True)
    # clf = LGBMRegressor(n_estimators=100)
    #elastic = LinearRegression()
    huber=HuberRegressor()
    # clf = Lasso()
    # clf = RidgeCV()
    crazy_clf = BayesianRidge()
    # clf = OrthogonalMatchingPursuit()
    clf = Ridge()
    # clf = GaussianProcessRegressor()
    # clf = LinearRegression()
    clf.fit(np.nan_to_num(period), temp_y)
    crazy_clf.fit(np.nan_to_num(period), temp_y)
    huber.fit(period, temp_y)
    
    normal_y = clf.predict(np.nan_to_num(anti_period))
    crazy_y = crazy_clf.predict(np.nan_to_num(anti_period))
    std = crazy_y.std()
    crazy_y = crazy_y * std * 2350
    final_y = (normal_y * normal_weight) + (crazy_y * crazy_weight)
    final_y = final_y.clip(low_y_cut, high_y_cut)
    score = r2_score(anti_y, final_y)
    scores.append(score)
    
    # print(timestamp)
    # if score > 0.000:
    clfs.append(clf)
    crazy_clfs.append(crazy_clf)
    hubers.append(huber)
    indexes.append(index)
    np.concatenate((period.std(axis=0), period.mean(axis=0)), axis=0)
    means.append(np.nan_to_num(np.concatenate((period.std(axis=0), period.mean(axis=0),[period.std(axis=1).mean(), period.mean(axis=1).mean()]), axis=0)))
    # elastics.append(elastic)
    # if timestamp % 100 == 0:
    # print(timestamp)

# for i in range(len(clfs)):
#     period = train[cols][~indexes[i]].copy()
#     temp_y = y[~indexes[i]]
#     mean_values = period.median(axis=0)
#     period.fillna(mean_values, inplace=True)
#     crazy_clf = crazy_clfs[i]
#     clf = clfs[i]
#     normal_y = clf.predict(np.nan_to_num(period))
#     crazy_y = crazy_clf.predict(np.nan_to_num(period))
#     std = crazy_y.std()
#     crazy_y = crazy_y * std * 2350
#     final_y = (normal_y * normal_weight) + (crazy_y * crazy_weight)
#     score = r2_score(temp_y, final_y)
print(scores, np.mean(np.array(scores)), np.std(np.array(scores)))
    
means = spatial.KDTree(means)
while True:
    target = observation.target
    test_std = np.array(observation.features[['technical_20','technical_30']].fillna(0).std()).reshape(1,-1)
    test = observation.features[cols].fillna(0)
    #median_values = test.median(axis=0)
    _, index = means.query(np.nan_to_num(np.concatenate((test.std(axis=0), test.mean(axis=0),[test.std(axis=1).mean(), test.mean(axis=1).mean()]), axis=0)), k=6)
    #test.fillna(median_values, inplace=True)
    #test.fillna(0, inplace=True)
    
    normal_y = clfs[index[0]].predict(np.nan_to_num(test)) + 0*clfs[index[1]].predict(np.nan_to_num(test))
    crazy_y = (crazy_clfs[index[0]].predict(np.nan_to_num(test))) + 0*(crazy_clfs[index[1]].predict(np.nan_to_num(test))) # + crazy_clfs[index[1]].predict(np.nan_to_num(test)))/2 #+ clfs[index[2]].predict(test) + clfs[index[3]].predict(test) + clfs[index[4]].predict(test))/5
    y_huber= hubers[index[0]].predict(np.nan_to_num(test))
    std = model_std.predict(np.nan_to_num(test_std))[0][0] 
    crazy_std= crazy_y.std()
    # print(std)
    crazy_y = crazy_y * ( ( 0.5 * std * 46) + (0.5 * crazy_std * 2350) )
    normal_y = normal_y * std * 62
    y_huber = y_huber * std * 39
    target.y = (1*(normal_y) + 4*(crazy_y) + 1*(y_huber))/6
    #target.y = (normal_y * normal_weight) + (crazy_y * crazy_weight)
    target.y = target.y.clip(low_y_cut,high_y_cut)
    
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break