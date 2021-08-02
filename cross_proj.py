import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Final_AMES.csv')
X = df.drop('SalePrice', axis = 1)
y = df['SalePrice']

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, train_size = 0.9, random_state = 101)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import ElasticNet
El_model = ElasticNet()
param_grid = {'alpha':[0.01,0.09,0.099,0.1,0.15, 0.3,0.5,0.9,0.99,1,10,25,50,100], 'l1_ratio':[0.01,0.09,0.099,0.1,0.15, 0.3,0.5,0.9,0.95,0.99,1]}

from sklearn.model_selection import GridSearchCV
gr_model = GridSearchCV(El_model, param_grid,cv = 15, scoring='neg_mean_absolute_error', verbose = 5)
gr_model.fit(X_train, y_train)
gr_model.best_estimator_
test = gr_model.cv_results_
test = pd.DataFrame(test)

y_pred = gr_model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error
mae_pred = mean_absolute_error(y_test, y_pred)
mse_pred = np.sqrt(mean_squared_error(y_test, y_pred))
 