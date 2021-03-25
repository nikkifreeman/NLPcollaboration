# Preprocess the Reback text messages for supervised biterm topic modeling
# Nikki Freeman
# 13 September 2020
# Last edited: 23 November 2020

# %% Load modules -------------------------------------------------------------
import pandas as pd 
import numpy as np 
import nltk
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import itertools
import random
import scipy

# %% Load the data ------------------------------------------------------------
# rawData = pd.read_excel('../../../Reback_TxtLibrary/Reback_Project Tech Support Text Message Library_NF.xlsx', \
#          sheet_name = 1, skiprows = 23, names = ['txtID', 'txtMsg'])

rawData = pd.read_excel('Reback_Project Tech Support Text Message Library_NF.xlsx', \
         sheet_name = 1, skiprows = 23, names = ['txtID', 'txtMsg'])         

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



# %% 
# Y = []
# txtMsgNum = []
# counter = 1
# for index, row in txtBiterm_df.iterrows():
#     if row['txtID'][0] == 'A':
#         draw = np.random.normal(loc = 1, scale = 1, size = 1)
#     else:
#         draw = np.random.normal(loc = 0, scale = 1, size = 1)
#     Y.append(draw[0])
#     txtMsgNum.append(counter)
#     counter += 1
# outcome = pd.DataFrame({'txtMsgNum':txtMsgNum, 'Y':Y})
# outcome.set_index('txtMsgNum')

# %% 
# Unique biterms
# allBiterms = [y for x in txtBiterms for y in x] # All biterms
# res = set()
# temp = [res.add((a, b)) for (a, b) in allBiterms if (a,b) and (b,a) not in res] # Remove duplicates
# del temp 

# Unique biterms and their constituent words
# w1 = []
# w2 = []
# [w1.append(a) for (a, b) in res]
# [w2.append(b) for (a, b) in res]
# b = [i for i in range(1, len(w2)+1)]
# allBiterms_df = pd.DataFrame(data = {'b':b, 'w1':w1, 'w2':w2}) # df version
# allBiterms_dict = dict(zip(res, b)) # dictionary version
# allBiterms = res # list of tuples version

# Add the biterm number to the text and biterm dataframe
# txtBiterm_df['b'] = ''
# for index, row in allBiterms_df.iterrows():
#     w1 = row['w1']
#     w2 = row['w2']
#     b = row['b']
#     b = b + 1
#     txtBiterm_df.loc[np.logical_and(np.logical_or(txtBiterm_df.w1 == w1, txtBiterm_df.w2 == w1), np.logical_or(txtBiterm_df.w1 == w2, txtBiterm_df.w2 == w2)), 'b'] = b +1
    
# %% Unique words -------------------------------------------------------------
allWords = [y for x in txtList for y in x]
allWords = np.array(allWords)
allWords = np.unique(allWords)
allWords = list(allWords)
n = [i for i in range(1, len(allWords)+1)]
allWords_dict = dict(zip(allWords, n))



# %% Set the initial topics ---------------------------------------------------
K = 3 # Number of topics
listOfTopics = list(range(1,K+1))
dfOfTopics = pd.DataFrame(listOfTopics, columns = ['topic'], index = listOfTopics)
initTopics = [random.sample(list(range(1, K+1)), 1) for i in allBiterms_df.index]
initTopics = list(itertools.chain.from_iterable(initTopics))

# Add the initial topics to the biterm dataframe
allBiterms_df['topic'] = initTopics

# Add a place for the topics in the txtBiterm dataframe
# bitermsAndInitTopics = allBiterms_df[['b', 'topic']]
# bitermsAndInitTopics = bitermsAndInitTopics.reset_index()
txtBiterm_df = txtBiterm_df.merge(allBiterms_df, on = ['b', 'w1', 'w2'])

# Create a dataframe to hold the counts for each biterm x topic
bitermTopicCount = pd.DataFrame(columns = ['topic' + str(i) for i in listOfTopics])
bitermTopicCount['b'] = allBiterms_df.b
bitermTopicCount = bitermTopicCount.fillna(0)

# Initial values for the supervision
eta_0 = [1 for i in range(0, K)]
Sigma_0 = np.identity(K)
sigma_0 = 1

# Place to hold durrent values of etas and sigmas
eta_current = eta_0
sigma2_current = sigma_0

# Place to store chains of etas and sigmas
sigma2s = [sigma_0]
etas = [np.array(eta_0)]

# Hyperparameters for the inverse gamma
nu_0 = 1
n = 657

# Convert outcomes to array for supervision
Y = outcome.Y.to_numpy()

# Little helper to make sure we have all of the topics in supervision
topicHelp = pd.DataFrame(columns = [i for i in range(1, K+1)])

# %%
Q = allBiterms_df.shape[0] # Number of biterms
L = 200 # Number of iterations
burn_in = 50

for l in range(0, L):
    print(l)
    for q in range(1, Q+1):
        w1 = allBiterms_df[allBiterms_df.b == q]['w1'].iloc[0]
        w2 = allBiterms_df[allBiterms_df.b == q]['w2'].iloc[0]

        # Calculate the sum of the biterms assigned to each topic
        n_k = allBiterms_df.loc[allBiterms_df.b != q, :].groupby('topic').count()[['b']]
        n_k = pd.merge(dfOfTopics, n_k, how = 'left', on = "topic") # In case not all topics are represented
        n_k = n_k.fillna(0)

        # Calculate the number of times the first word of the biterm is assigned to each topic
        n_kw1 = allBiterms_df[np.logical_or(allBiterms_df.w1 == w1, allBiterms_df.w2 == w1)].groupby('topic').count()[['b']]
        n_kw1 = pd.merge(dfOfTopics, n_kw1, how = "left", on = "topic")
        n_kw1 = n_kw1.fillna(0)

        # Calculate the number of times the second word of the biterm is assigned to each topic
        n_kw2 = allBiterms_df[np.logical_or(allBiterms_df.w1 == w2, allBiterms_df.w1 == w2)].groupby('topic').count()[['b']]
        n_kw2 = pd.merge(dfOfTopics, n_kw2, how = "left", on = "topic")
        n_kw2 = n_kw2.fillna(0)

        # Calculate the sum of the words in biterms assigned to each topic
        n_kw12 = allBiterms_df[np.logical_or(np.logical_or(allBiterms_df.w1 == w1, allBiterms_df.w1 == w2), np.logical_or(allBiterms_df.w2 == w1, allBiterms_df.w2 == w2))].groupby('topic').count()[['b']]
        n_kw12 = pd.merge(dfOfTopics, n_kw12, how = "left", on = "topic")
        n_kw12 = n_kw12.fillna(0)

        # Prepare the supervision piece
        txtsWithq = txtBiterm_df.txtMsgNum[txtBiterm_df.b == q].to_list()
        X = txtBiterm_df[txtBiterm_df.txtMsgNum.isin(txtsWithq)]

        # Calculate the probabilities
        probs = np.zeros(K)
        for i in range(0, K):
            # Supervision piece
            # X.topic.loc[X.b == q] = i+1
            X.loc[X.b == q, 'topic'] = i+1
            x = X.groupby(['txtMsgNum', 'topic', 'Y']).count()
            x.reset_index(inplace = True)
            x = x.pivot(index = 'Y', columns = 'topic', values = 'b')
            x = topicHelp.append(x)
            x = x.fillna(0)
            x = x.div(x.sum(axis = 1), axis = 0)
            y = x.index.to_numpy()
            x = x.to_numpy()
            linPred = np.matmul(x, eta_current)
            pr_y = 1
            for j in range(0, len(y)):
                pr_y = pr_y * scipy.stats.norm.pdf(y[j], loc = linPred, scale = sigma2_current)
        
            denom = n_kw12.iloc[i, ][0] + 1
            denom = denom**2
            probs[i] = (pr_y[0] * n_k.iloc[i, ][0] + 1) * (n_kw1.iloc[i, ][0] + 1) * (n_kw2.iloc[i, ][0] + 1) * (1/denom)

        # Normalize the probabilities
        probs = probs/sum(probs)

        # Draw the new topic for the biterm
        new_topic = np.random.choice(listOfTopics, size = 1, p = probs)[0]

        # Put the new topic in the dataframe
        allBiterms_df.loc[allBiterms_df.b == q, 'topic'] = new_topic
        txtBiterm_df.loc[txtBiterm_df.b == q, 'topic'] = new_topic

        # Update the counter
        if l >= burn_in:
            colName = 'topic' + str(new_topic)
            bitermTopicCount.loc[bitermTopicCount.b == q, colName] += 1

    # Update supervision parameters
    temp = txtBiterm_df.groupby(['txtMsgNum', 'topic']).count()
    temp.reset_index(inplace = True)
    temp = temp.loc[:, ['txtMsgNum', 'topic', 'b']]
    temp = temp.pivot(index = 'txtMsgNum', columns = 'topic', values = 'b')
    # txtMsgNum = [i for i in range(1, outcome.Y.size + 1)]
    # txtMsgNum = pd.DataFrame({'txtMsgNum':txtMsgNum})
    # temp = txtMsgNum.join(temp, on = 'txtMsgNum', how = 'left')
    temp = temp.fillna(0)
    # temp = temp.drop(['txtMsgNum'], axis = 1)
    temp = temp.div(temp.sum(axis = 1), axis = 0)

    # Y =  np.delete(Y, [398, 478, 557])

    # Calculate V and m
    X = temp.to_numpy()
    V = Sigma_0 + np.matmul(X.transpose(),X)*(1/sigma2_current)
    V = np.linalg.inv(V)
    m = np.matmul(Sigma_0, eta_current) + np.matmul(X.transpose(), Y)*(1/sigma2_current) 
    m = np.matmul(V, m)

    # Update eta
    eta_current = np.random.multivariate_normal(mean = m, cov = V)
    etas.append(eta_current)

    # Calculate SSR(eta)
    # SSR_eta = np.matmul(Y.transpose(), Y) 
    # SSR_eta = SSR_eta * -2*np.matmul(np.matmul(eta_current.transpose(), X.transpose()), Y)
    # SSR_eta = SSR_eta + np.matmul(np.matmul(eta_current.transpose(), X.transpose()), np.matmul(X, eta_current))

    # Update sigma2
    # shape = (nu_0 + n)/2
    # scale = (nu_0*sigma_0 + SSR_eta)*(1/2)
    # sigma2_current = scipy.stats.invgamma.rvs(shape, scale, 1)
    # sigma2_current = np.absolute(sigma2_current)
    # sigma2s.append(sigma2_current)
    sigma2_current = 1



    
# %% save output --------------------------------------------------------------
bitermTopicCount.to_csv('bitermTopicCount.csv')
allBiterms_df.to_csv('allBiterms_df.csv')
# %%
pd.melt(bitermTopicCount, id_vars = 'b', value_vars = ['topic1', 'topic2', 'topic3']).groupby('b').max()
# %%




# %%
