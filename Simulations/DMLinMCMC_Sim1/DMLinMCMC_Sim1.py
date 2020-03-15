import numpy as np 
import pandas as pd 
import random 
import scipy.stats 

# Load the document-text matrix
DTMPath = '/Users/Nikki/Dropbox/UNC/Causal NLP/Reback_TxtLibrary/RebackDTM.csv'
fullDTM = pd.read_csv(DTMPath, index_col= 'msgID')

# Set the number of topics
K = 2

# Subset on the text messages for this simulation
DTM = fullDTM.loc['A1a001':'B3b017', :]
# # Remove the rows that sum to 0 (these are messages that are all stop words)
# DTM = DTM.loc[(DTM.sum(axis = 1) > 5), :]
# print(DTM)
# Remove the columns that are only 0s (those words don't show up in the subset of text messages)
DTM = DTM.loc[:, DTM.sum() > 0] #112 text messages and 195 words


# Create ppts and assign a text message
random.seed(1000)
R = 112 # Number of participants
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

# Assign the words to numbers (1:N); put in a dictionary
N = len(DTM.columns)
wordDict = dict(zip(DTM.columns, range(1,N+1)))
DTMLong = pd.melt(DTM.reset_index(), id_vars = 'msgID')
DTMLong = DTMLong[DTMLong.value > 0]
DTMLong = DTMLong.drop(columns = ['value'])
# Add the word numbers to the text message long df
DTMLong['wordNum'] = DTMLong['variable'].map(wordDict)
DTMLong = DTMLong.rename(columns = {'variable':'word'})

# Each text is a 2D array of words and topics
# 2D arrays live in a list of arrays
# Initialize the topics
# msgIDs = DTMLong.msgID.unique()


# txtWordTopicList =[]
# for msg in msgIDs:
#     # label
#     wordList = DTMLong[DTMLong['msgID'] == msg]
#     wordList = wordList['wordNum'].tolist()
#     N_m = len(wordList)
#     topicList = np.random.choice(range(1, K+1), size = N_m, replace = True)
#     topicList = list(topicList)
#     msgArray = np.asarray([wordList, topicList])
#     txtWordTopicList = txtWordTopicList.append(list(msg, msgArray))
# print(txtWordTopicList)

# Assign the messages an id from 1 to M
msgIDs = DTMLong.msgID.unique()
M = len(list(msgIDs))
msgDict = dict(zip(msgIDs, range(1, M+1)))
# Add the message numbers to the text message long df
DTMLong['msgNum'] = DTMLong['msgID'].map(msgDict)

# Merge the participant dataframe to the long dtm data frame
DTMLong = DTMLong.merge(pptDF, how = 'left', left_on = 'msgID', right_on = 'A')
DTMLong = DTMLong.drop(columns = 'A')

# Get lists of the labeled messages and the unlabeled messages
unlabeledMsgs = DTMLong.msgID[DTMLong['Y'].isnull()].unique().tolist()
labeledMsgs = DTMLong.msgID[DTMLong['Y'].notnull()].unique().tolist()

# Assign topics
DTMLong['topic'] = np.random.choice(range(1, K+1), size = DTMLong.shape[0], replace = True)

# Hyperparameters
alpha = np.repeat(1, K)
delta = np.repeat(1, K)
Sigma_0_inv = np.identity(K)
beta_0 = np.array([[1], [1]])
nu_0 = 3
sigma_0 = 0.5

L = 500
Zchain = pd.DataFrame()
betaChain = [np.array([0, 0])]
sigma2 = [np.array([1])]

for l in range(0, L):
    DTMLong_new = pd.DataFrame(DTMLong)
    # Divide the messages in half
    # Make sure that there is a balance of labeled data in each set
    unlabeledToTrain = np.random.choice(unlabeledMsgs, size = round((M - R)/2), replace = False)
    unlabeledToTrain = unlabeledToTrain.tolist()
    unlabeledToTest = list(set(unlabeledMsgs) - set(unlabeledToTrain))
    labeledToTrain = np.random.choice(labeledMsgs, size = round(R/2), replace = False)
    labeledToTrain = labeledToTrain.tolist()
    labeledToTest = list(set(labeledMsgs) - set(labeledToTrain))
    trainingMsgIds = unlabeledToTrain + labeledToTrain
    # No need for testing MsgIDs because only the labeled ones are used in estimation

    # Update the regression parameters
    ## We only use the labeled cases for the update
    labeledTrainingCases = DTMLong_new[DTMLong_new['msgID'].isin(labeledToTrain)] 
    temp = labeledTrainingCases.groupby(['msgNum', 'topic']).count()
    temp = temp.reset_index()
    temp = temp.pivot(index = 'msgNum', columns = 'topic', values = 'msgID')
    temp = temp.fillna(0)
    temp['sumTopics'] = temp.sum(axis = 1)
    temp['topic1'] = temp[1]/temp['sumTopics']
    temp['topic2'] = temp[2]/temp['sumTopics']
    X = temp[['topic1', 'topic2']] # Hang on to this for the supervision piece later
    X_values = X.to_numpy()
    y = labeledTrainingCases[['msgID', 'Y']].drop_duplicates()
    y = y['Y'].to_numpy()
    y = np.reshape(y, (-1, 1))
    ## Update beta
    ### Compute V and m
    V = np.linalg.inv(Sigma_0_inv + np.dot(X_values.transpose(), X_values)/sigma2[l])
    m_beta = np.dot(V, np.dot(Sigma_0_inv, beta_0) + np.dot(X_values.transpose(), y))
    m_beta = m_beta.flatten()
    ### Update beta
    beta_new = np.random.multivariate_normal(m_beta, V)
    betaChain.append(beta_new)
    ### Compute SSR(beta)
    SSR_beta = np.dot(y.transpose(), y) - 2*np.dot(np.dot(betaChain[l+1].transpose(), X_values.transpose()), y) + np.dot(np.dot(betaChain[l+1].transpose(), X_values.transpose()), np.dot(X_values, betaChain[l+1]))
    ig_a = (nu_0 + round(R/2))/2
    ig_b = (nu_0 * sigma_0 + SSR_beta)/2
    sigma2.append(scipy.stats.invgamma.rvs(ig_a, ig_b, size = 1))
    

    
    for index, row in DTMLong.iterrows():
        if row['msgID'] not in trainingMsgIds:
            continue
        m_star = row['msgNum']
        n_star = row['wordNum']
    
        # Replace the topic with a zero so it's not included
        DTMLong_new.at[(DTMLong_new['msgNum'] == m_star )& (DTMLong_new['wordNum'] == n_star), 'topic'] = 0

        # Calculate n_hat
        n_hat = list()
        n_hatList = DTMLong_new.topic[DTMLong_new['msgNum'] == m_star].tolist()
        n_hatArray = np.asarray(n_hatList)
        for k_tilde in range(1, K+1):
            sumK_tilde = sum(n_hatArray == k_tilde)
            n_hat.append(sumK_tilde)
        
        # Calculate n_tilde
        n_tilde = list()
        n_tildeList = DTMLong_new.topic[DTMLong_new['wordNum'] == n_star].tolist()
        n_tildeArray = np.asarray(n_tildeList)
        for k_tilde in range(1, K+1):
            sumK_tilde = sum(n_tildeArray == k_tilde)
            n_tilde.append(sumK_tilde)
    
        # Calculate the probs
        ## a is the prob associated with n_tilde
        a_top = n_tilde + delta
        a_bottom = sum(n_tilde + delta)
        a = a_top/a_bottom
    
        ## b is the prob associated with n_hat
        b_top = n_hat + alpha
        b_bottom = sum(n_hat + alpha)
        b = b_top/b_bottom

        # If there is supervision calculate c, else calculation the probs using a and b
        if(DTMLong_new.Y[DTMLong_new['msgNum'] == m_star].isnull().values.any()):
            # Calculate the probs
            probs = a*b/(sum(a*b))
        else:
            z_bar = [len(DTMLong_new.Y[(DTMLong_new['msgNum'] == m_star) & (DTMLong_new['topic'] == topic)]) for topic in range(1, K+1)]
            if sum(z_bar) == 0:
                # For the rare case when there is only one word in the message that isn't a stop word
                probs = a*b/sum(a*b)
            else:
                z_bar = [count/sum(z_bar) for count in z_bar]
                z_bar = np.asarray(z_bar)
                y = DTMLong_new.Y[DTMLong_new['msgNum'] == m_star].values
                ## c is the prob associated with the supervision
                c = scipy.stats.norm.pdf(y[0], loc = sum(z_bar * betaChain[-1]), scale = sigma2[-1])
                ## Calculate the probs
                probs = a*b*c
                probs = probs/sum(probs)


        # Draw a new z
        z_new = np.random.choice(range(1, K+1), size = 1, p = probs)
        DTMLong_new.at[(DTMLong_new['msgNum'] == m_star )& (DTMLong_new['wordNum'] == n_star), 'topic'] = z_new
    
    # Label the sample
    DTMLong_new['sample'] = l+1
    Zchain = Zchain.append(DTMLong_new)
    DTMLong = DTMLong_new

Zchain.to_csv('Zchain.csv')
pd.DataFrame(betaChain).to_csv('betaChain.csv')
pd.DataFrame(sigma2).to_csv('sigma2.csv')









    



    







