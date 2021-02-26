#%%
import numpy as np 
import tensorflow as tf 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
#######################################################################################################
# building a system for ranking customer tickets by priority and routing them to appropriate department 
# The model has three inputs 
# 1. The title of the ticket. 
# 2. The text body of ticket. 
# 3. any tags added by the user.  

# the model has two outputs 
# --> The priority score of ticket [sigmoid] (0.......1) 
# --> The department that should handel the ticker (softmax over the set of department)
# %%
