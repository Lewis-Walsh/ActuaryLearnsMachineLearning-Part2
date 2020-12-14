# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 08:36:09 2020

@author: walshl
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:20:45 2020

@author: walshl
"""

#%% Do imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

#%% Load test and train data
        
train_data = pd.read_csv("C:\\Work\\Machine Learning experiments\\Kaggle\\Titanic\\train2.csv")
train_data.head()

test_data = pd.read_csv("C:\\Work\\Machine Learning experiments\\Kaggle\\Titanic\\test2.csv")
test_data.head()



#%% Set-up model

Y = train_data["Survived"]
features = ["Pclass","Sex","SibSp","Parch","Embarked","AgeBand"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])


#%% Run model
RFmodel = RandomForestClassifier(n_estimators=100,random_state=1)
RFmodel.fit(X,Y)
predictions = RFmodel.predict(X_test)


#%% Output

output = pd.DataFrame({'PassengerID': test_data.PassengerId, 'Survived': predictions})
output.to_csv('Testoutput.csv',index=False)
