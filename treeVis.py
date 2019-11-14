#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 18:45:18 2019

@author: sohel
"""

import pandas as pd
import numpy as np
from sklearn import tree


# install pydotplus and graphviz
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  



class vis(object):
	
	"""
	This class contains visualisations for machine learning algorithms	
	"""

	def __init__(self):
		pass


	def dtree_visual(self,treeClf):




		dot_data = StringIO()
		export_graphviz(treeClf, out_file=dot_data,  
						filled=True, rounded=True,
						special_characters=True)
		graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
		return Image(graph.create_png())


	def __del__(self):
		pass
