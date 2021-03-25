# Preprocess the Reback text messages for supervised biterm topic modeling
# Nikki Freeman
# 13 September 2020
# Last edited: 1 January 2021

# %% Load modules -------------------------------------------------------------
import pandas as pd 
import numpy as np 
import scipy
import nltk
import statistics
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import itertools
import random
import scipy

# %% Load the data ------------------------------------------------------------
rawData = pd.read_excel('../../../Reback_TxtLibrary/Reback_Project Tech Support Text Message Library_NF.xlsx', \
         sheet_name = 1, skiprows = 23, names = ['txtID', 'txtMsg'])

# rawData = pd.read_excel('Reback_Project Tech Support Text Message Library_NF.xlsx', \
#          sheet_name = 1, skiprows = 23, names = ['txtID', 'txtMsg'])         

# %% Pre-processing steps -----------------------------------------------------
# Remove empty rows from the raw data
rawData = rawData.dropna()

# Convert the text to all lower case
rawData.txtMsg = rawData.txtMsg.str.lower()

# Remove numbers
rawData.txtMsg = rawData.txtMsg.str.replace('[0-9]', '', regex = True)

# Remove punctuation
rawData.txtMsg = rawData.txtMsg.str.replace('/', " ")
rawData.txtMsg = rawData.txtMsg.str.replace('[.,-?()+=_%:"!$]', '', regex = True)

# Deal with special words
rawData.txtMsg = rawData.txtMsg.str.replace("hep C", 'hepc')
rawData.txtMsg = rawData.txtMsg.str.replace("america's", "america")
rawData.txtMsg = rawData.txtMsg.str.replace("barebacking", "bareback")
rawData.txtMsg = rawData.txtMsg.str.replace("bootybumps", "bootybump")
# rawData.txtMsg = rawData.txtMsg.str.replace("brings", "bring")
# rawData.txtMsg = rawData.txtMsg.str.replace("brushed", "brush")

# Remove stop words
txtList = rawData['txtMsg'].tolist()
txtList = [x.split() for x in txtList]

stopWords = ["a", "his", "will", "with", "your", "be", "to", "yourself"\
    "N", "some", "are", "aren't", "not", "you're", "if", "don't", "what",
    "and", "too", "doesn't", "of", "no", "is", "isn't", "for", "can't",
    "he's", "she's", "at", "it", "pm", "am", "just", "that", "but",
    "about", "could", "couldn't", "on", "or", "in", "him", "her", "you",
    "do", "can", "then", "there", "up", "down", "as", "had", "got", "the",
    "all", "it's", "he", "she", "someone", "than", "when", "that's",
    "ok", "everybody's", "it's", "won't", "get", "by", "ain't",
    "aka", "an", "been", "can't", "he'll", "she'll", "it's", "what's"]
 
ls2 = []
for txt in txtList:
    ls1 = []
    for word in txt:
        if(word not in stopWords):
            ls1.append(word)
    ls2.append(ls1)
txtList = ls2
del ls2, ls1

# Lemmatization
txtList = [[WordNetLemmatizer().lemmatize(y) for y in x] for x in txtList ]

# Remove duplicate terms in a single text
txtArray = [np.array(x) for x in txtList]
txtArray = [np.unique(x) for x in txtList]
txtList = [list(x) for x in txtList]



# %% Create the biterms -------------------------------------------------------
# List of texts and their biterms
txtBiterms = [list(itertools.combinations(x, 2)) for x in txtList] 

# Create a dataframe to link biterms back to text messages
counter = 1
txtBiterm_df = pd.DataFrame(columns = ['txtMsgNum', 'w1', 'w2', 'txtID'])
for i in txtBiterms:
    w1 = [j[0] for j in i]
    w2 = [j[1] for j in i]
    txtMsgNum = [counter for j in i]
    txtID = rawData.txtID.iloc[counter - 1]
    txtID = [txtID for j in i]
    temp = pd.DataFrame([txtMsgNum, w1, w2, txtID]).transpose()
    temp.columns = ['txtMsgNum', 'w1', 'w2', 'txtID']
    txtBiterm_df = txtBiterm_df.append(temp, ignore_index=True)
    counter += 1


# Unique biterms and biterm number
allBiterms_df = txtBiterm_df[['w1', 'w2']].drop_duplicates()
b = [i for i in range(1, allBiterms_df.shape[0] + 1)]
allBiterms_df['b'] = b

# Add the biterm number to the text and biterm data frame
txtBiterm_df = txtBiterm_df.merge(allBiterms_df, how = 'left', on = ['w1', 'w2'])

# %% Get a list of the unique words -------------------------------------------
allWords = [y for x in txtList for y in x]
allWords_array = np.array(allWords)
allUniqueWords_array = np.unique(allWords_array)
allUniqueWords_list = allUniqueWords_array.tolist()

# Number the words
allUniqueWords_df = pd.DataFrame({'word':allUniqueWords_list, 'word_num':[i for i in range(1, len(allUniqueWords_list) + 1)]})

# Add word_num to allBiterms
allBiterms_df = allBiterms_df.merge(allUniqueWords_df, how = 'left', left_on = 'w1', right_on = 'word')
allBiterms_df = allBiterms_df.rename(columns = {'word_num':'w1_num'})
allBiterms_df = allBiterms_df.drop(columns = 'word')
allBiterms_df = allBiterms_df.merge(allUniqueWords_df, how = 'left', left_on = 'w2', right_on = 'word')
allBiterms_df = allBiterms_df.rename(columns = {'word_num':'w2_num'})
allBiterms_df = allBiterms_df.drop(columns = 'word')

# %% Create the outcomes for the supervision ----------------------------------
Y = []
txtID = txtBiterm_df.txtID.unique()
txtBiterm_df['Y'] = 0

for i in txtID:
    if i[0] == 'A':
        draw = np.random.normal(loc = 1, scale = 1, size = 1)
    else: 
        draw = np.random.normal(loc = 0, scale = 1, size = 1)
    # txtBiterm_df.Y[txtBiterm_df.txtID == i] = draw
    txtBiterm_df.loc[txtBiterm_df.txtID == i, 'Y'] = draw
    
outcome = txtBiterm_df[['txtMsgNum', 'Y']]
outcome = outcome.drop_duplicates()
outcome.set_index('txtMsgNum')

# txtBiterm_df.drop(columns=['txtID', 'Y'])

# %% Initialize parameters
K = 2 # Number of topics

# Initial \xi_{r, k}
xi_init = pd.DataFrame(allBiterms_df)
for i in range(1, K+1):
    colName = "k" + str(i)
    xi_init[colName] = [1/len(allBiterms_df.b) for j in range(1, len(allBiterms_df.b) + 1 )] 

# Initial \phi_{n,k}
phi_init = pd.DataFrame(allUniqueWords_df)
for i in range(1, K+1):
    colName = "k" + str(i)
    phi_init[colName] = [1/len(allUniqueWords_df.word) for j in range(1, len(allUniqueWords_df.word) + 1)]

# Initial \gamma_k
gamma_init = pd.DataFrame({'k':[j for j in range(1, K+1)], 'gamma_k':[1 for j in range(1, K+1)]})

# Initial \sigma^2
sigma2_init = 1

# Initial \eta
eta_init = pd.DataFrame({'k':[j for j in range(1, K+1)], 'eta_k':[1 for j in range(1, K+1)]})

# Initialize alpha
alpha = [1/K for i in range(1, K+1)]

# %% E-step -------------------------------------------------------------------
phi_current = phi_init.copy(deep = True)
gamma_current = gamma_init.copy(deep = True)
sigma2_current = sigma2_init
eta_current = eta_init
xi_current = xi_init.copy(deep = True)

iter = 1
tol = 0.0001
delta = 1

while delta >tol:
    if iter == 1000:
        break
    # Update x_rk
    xi_old = xi_current.copy(deep = True)
    for k_star in range(1, K+1):
        for r_star in range(1, len(allBiterms_df.b) + 1):
            # Words that correspond to the biterm r_star
            word1 = int(allBiterms_df.w1_num[allBiterms_df.b == r_star])
            word2 = int(allBiterms_df.w2_num[allBiterms_df.b == r_star])

            # y's the correspond to the biterm r_star
            texts = txtBiterm_df.txtMsgNum[txtBiterm_df.b == r_star].tolist()
            y = outcome.Y[outcome['txtMsgNum'].isin(texts)]

            # Other biterms in texts that contain b_rstar
            otherBiterms = txtBiterm_df.b[txtBiterm_df['txtMsgNum'].isin(texts)].tolist()
            otherBiterms = [i for i in otherBiterms if i is not r_star]

            # eta_{k^\ast}
            eta_kstar = float(eta_current.eta_k[eta_current.k == k_star])

            # Last piece of sum is complicated. Calculate it here
            lastPiece = 0
            for k in range(1, K+1):
                if k == k_star:
                    continue
                colName1 = "k" + str(k)
                temp = sum(xi_current[colName1][xi_current['b'].isin(otherBiterms)])
                if np.isnan(temp):
                    temp = 0
                lastPiece = lastPiece + (temp + \
                    xi_current[colName1][xi_current['b'] == r_star]) *\
                        float(eta_current.eta_k[eta_current.k == k_star]) * float(eta_current.eta_k[eta_current.k == k])
            lastPiece = lastPiece * (1/sigma2_current) *(1/len(y)**2)
            
            


            colName = "k" + str(k_star)
            temp2 = sum(xi_current[colName][xi_current['b'].isin(otherBiterms)])
            if np.isnan(temp2):
                temp2 = 0
            update = float(np.log(phi_current[colName][phi_current.word_num == word1])) +\
                float(np.log(phi_current[colName][phi_current.word_num == word2])) +\
                    scipy.special.digamma(float(gamma_current.gamma_k[gamma_current.k == k_star])) -\
                        scipy.special.digamma(float(sum(gamma_current.gamma_k))) + \
                            (1/sigma2_current)*statistics.mean(y)*eta_kstar +\
                                (1/(2 *sigma2_current))*(eta_kstar**2)*(1/(len(y)**2))*(len(y) + 2*temp2) +\
                                    lastPiece
            update = np.exp(update)
            xi_current.loc[xi_current.b == r_star, colName] = update

    # Normalize the xi        
    colNames = ["k" + str(i) for i in range(1, K+1)]
    normalizer = xi_current.loc[:, colNames].sum(axis = 1)
    for col in colNames:
        xi_current[col] = xi_current[col].divide(normalizer) 

    # Update gamma
    gamma_old = gamma_current.copy(deep = True)
    for k in range(1, K+1):
        colName = "k" + str(k)
        gamma_current.loc[gamma_current.k == k, 'gamma_k'] = alpha[k-1] + sum(xi_current[colName])

    delta_gamma = max(abs(gamma_current.gamma_k - gamma_old.gamma_k))
    delta_xi = abs(xi_old[colNames] - xi_current[colNames])
    delta_xi = max(delta_xi.max())
    delta = max(delta_gamma, delta_xi)
    print(iter)
    print(delta)
    iter += 1



        


        
        
        



# %%
