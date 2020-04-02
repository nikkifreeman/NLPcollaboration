import numpy as np 
import pandas as pd 
import random 

# Load the document-text matrix
DTMPath = '/Users/Nikki/Dropbox/UNC/Causal NLP/Reback_TxtLibrary/RebackDTM.csv'
fullDTM = pd.read_csv(DTMPath, index_col= 'msgID')

# Subset on the text messages for this simulation
DTM = fullDTM.loc['A1a001':'B3b017', :]
# Remove the columns that are only 0s (those words don't show up in the subset of text messages)
DTM = DTM.loc[:, DTM.sum() > 0] #112 text messages and 195 words

# Create ppts and assign a text message
random.seed(1001)
R = 60 # Number of participants
ppt = list(range(1, R+1))
assignments = np.random.choice(list(DTM.index), size = R, replace = False)
pptList = [ppt, assignments]
pptDF = pd.DataFrame(pptList).transpose()
pptDF = pptDF.rename(columns = {0:'pptID', 1:'A'})

# Create the outcomes
pptDF['Y'] = 0
mu_y_a = 10
sigma_y_a = 1
mu_y_b = 5
sigma_y_b = 1
pptDF['topicA'] = pptDF['A'].str.contains('^A')
pptDF['Y'] = pptDF['topicA']. apply(lambda x: np.random.normal(mu_y_a, sigma_y_a) if x == True else np.random.normal(mu_y_b, sigma_y_b))
print(pptDF)

# Functions for DML in MCMC
print(DTM.columns)


