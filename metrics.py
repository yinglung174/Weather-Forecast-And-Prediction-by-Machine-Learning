#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 11:11:14 2019

@author: sohel
"""

from sklearn.metrics import accuracy_score

class matrix(object):
    
    def __init__(self):
        pass
    
    def accuracy(self,x,y):
        acc=accuracy_score(x,y)
        return(acc)
        
    def __del__(self):
        pass
    