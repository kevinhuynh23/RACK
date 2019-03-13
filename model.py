#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor    # KNN regressor
from sklearn.preprocessing import StandardScaler      # scaling data
from sklearn.pipeline import make_pipeline 

from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings("ignore")


# In[2]:


listing = pd.read_csv('data/prepped/lis_out.csv')


# In[3]:


listing.head()


# In[4]:


#variables to use: weekofyear, city, all temp variables, precip_per_sq_km

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


# In[5]:


listing.corr().price_float.sort_values()


# In[6]:


selected_cols = ['price_float',
                'accommodates',
                'beds',
                'bedrooms',
                'cf',
                'sd',
                'zipcode_code',
                'host_is_super_host',
                'bathrooms',
                'space',
                'minimum_nights',
                'latitude',
                'host_listings_count',
                'review_scores_rating',
                'name',
                'longitude',
                'review_true_score', 'host_response_rate_float']

temp = listing[selected_cols]


# In[7]:


mask = np.zeros_like(temp.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12, 12))

cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax.set_title("Correlation Map of Current Features for Airbnb")
corr_map = sns.heatmap(temp.corr(), mask=mask, cmap=cmap, vmax=1.0, center=0, square=True, linewidths=.5, cbar_kws={"shrink":.5})

plt.show()


# In[8]:


train_features, test_features, train_outcome, test_outcome = train_test_split(
    temp.drop('price_float', axis=1),
    temp.price_float,
    test_size=0.30,
    random_state=15
)


# ## Baseline Linear Regression

# In[9]:


ls = LinearRegression()
ls.fit(train_features, train_outcome)
pred = ls.predict(test_features)


# In[10]:


base_lin_score = mean_absolute_error(test_outcome, pred)
base_lin_score


# In[11]:


fig_blr = plt.figure(figsize=(8,8))
plt.scatter(test_outcome, pred)
plt.plot([0,350], [0, 350], color='k')
plt.title('Baseline Linear Regression')
plt.xlabel('Test Outcome')
plt.ylabel('Test Predictions')
plt.show()


# ## Ridge Linear Regression

# In[12]:


lin_reg = Ridge()

std_scaler = StandardScaler()

# Define a pipeline that uses your scaler and classifier
lin_pipe = make_pipeline(std_scaler, lin_reg)

# Define a grid to search through
param_grid_lin = {'ridge__alpha':[1],
                  'ridge__fit_intercept':[True,False],
                  'ridge__normalize':[True,False],
                  'ridge__copy_X':[True, False],
                  'ridge__solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

# Perform a  grid search of your pipeline
grid_lin = GridSearchCV(lin_pipe, param_grid_lin, scoring="neg_mean_absolute_error")


# In[13]:


grid_lin.fit(train_features, train_outcome)


# In[14]:


ridge_score = grid_lin.score(test_features, test_outcome)
ridge_score


# In[15]:


fig_rlr = plt.figure(figsize=(8,8))
plt.scatter(test_outcome, grid_lin.predict(test_features))
plt.plot([0,350], [0, 350], color='k')
plt.title('Ridge Linear Regression')
plt.xlabel('Test Outcome')
plt.ylabel('Test Predictions')
plt.show()


# ## Negative Binomial Regression

# In[16]:


formula = ' + '.join(i for i in list(selected_cols[1:]))
formula = 'price_float ~ ' + formula


# In[17]:


mod = smf.glm(formula=formula,
              data=temp,
              family=sm.families.NegativeBinomial())
mod = mod.fit()

predictions_nb = mod.predict(test_features)

nbr_score = mean_absolute_error(predictions_nb, test_outcome)
nbr_score


# In[18]:


fig_nbr = plt.figure(figsize=(8,8))
plt.scatter(predictions_nb, test_outcome)
plt.plot([0,350], [0, 350], color='k')
plt.title('Negative Binomial Regression')
plt.xlabel('Test Outcome')
plt.ylabel('Test Predictions')
plt.show()


# ## K-Nearest Neighbors

# In[19]:


#KNN Regression Model
knn_reg = KNeighborsRegressor()

std_scaler = StandardScaler()

# Define a pipeline that uses your scaler and classifier
pipe_knn = make_pipeline(std_scaler, knn_reg)

# Define a grid to search through
param_grid_knn = {'kneighborsregressor__n_neighbors':range(10, 20), 
                  'kneighborsregressor__weights':["uniform", "distance"],
                  'kneighborsregressor__algorithm':['auto']}

# Perform a  grid search of your pipeline
grid_knn = GridSearchCV(pipe_knn, param_grid_knn, cv=10, scoring="neg_mean_absolute_error")


# In[20]:


grid_knn.fit(train_features, train_outcome)


# In[21]:


knn_score = grid_knn.score(test_features, test_outcome)
knn_score


# In[22]:


fig_knn = plt.figure(figsize=(8,8))
plt.scatter(test_outcome, grid_knn.predict(test_features))
plt.plot([0,350], [0, 350], color='k')
plt.title('K-Nearest Neighbors Regression')
plt.xlabel('Test Outcome')
plt.ylabel('Test Predictions')
plt.show()


# ## XGBooster Regression

# In[23]:


xgb_model = XGBRegressor()

param_grid_xgb = {'n_jobs':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [5],
              'min_child_weight':[1],
              'colsample_bytree': [0.8],
              'gamma':[0.5],
              'subsample':[0.9],
              'random_state':[11],
              'n_estimators': [1000] #number of trees
              }

grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, scoring='neg_mean_absolute_error')


# In[24]:


grid_xgb.fit(train_features, train_outcome)


# In[25]:


xgb_score = grid_xgb.score(test_features, test_outcome)
xgb_score


# In[26]:


fig_xbg = plt.figure(figsize=(8,8))
plt.scatter(test_outcome, grid_xgb.predict(test_features))
plt.plot([0,350], [0, 350], color='k')
plt.title('XG Booster Regression')
plt.xlabel('Test Outcome')
plt.ylabel('Test Predictions')
plt.show()

