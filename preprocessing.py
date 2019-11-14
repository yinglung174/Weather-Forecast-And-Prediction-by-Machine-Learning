#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 10:00:43 2019

@author: sohel
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

class scaler(object):
    
    def __init__(self):
        pass
    
    def scalling(self,x):
        sc=MinMaxScaler(feature_range=(0,1))
        scaled_data=sc.fit_transform(x)
        return(scaled_data)
        
        
    def __del__(self):
        pass
   
class splitter(object):
    
    def __init__(self):
        pass
    
    def decomposition(self,x,y,test_size=0.25,random_state=0):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=random_state)
        return(x_train,x_test,y_train,y_test)
        
    def __del__(self):
        pass


class Encoder(object):
    def __init__(self):
        pass
    
    def labelEncoding(self,x):
        le=LabelEncoder()
        x=le.fit_transform(x)
        return(x)
        
    def one_hot_encoder(self,y,cf):
        ohe=OneHotEncoder(categorical_features=[cf])
        ohe=OneHotEncoder(categorical_features=cf)
        y=ohe.fit_transform(y).toarray()
        return(y)
        
    def __del__(self):
        pass
        
    





















