"""
Experiment 6
"""
# Import modules ------------------------------------------------------
# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import os
import torch
from torch.nn import Parameter
import pyro 
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.ops.stats as stats

# Load the text messages and put into a list for transformation -------
# %%
# Read in the text messages
texts = pd.read_excel(
    '../../../Reback_TxtLibrary/Reback_Project Tech Support Text Message Library.xlsx',
    sheet_name = "Library",
    names = ["msgID", "msg"],
    skiprows = 23)

# Drop the NA rows
texts = texts.dropna()

# Create a list of ids and a list of messages
msgID_list = texts['msgID'].tolist()
msg_list = texts['msg'].tolist()

# Embed the texts into a pretrained model -----------------------------
# %%
model = SentenceTransformer('bert-base-nli-mean-tokens')
msg_embeddings = model.encode(msg_list)

# %%
# Take a peek at the embeddings
for sentence, embedding in zip(msg_list, msg_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

# Fit the LVGP --------------------------------------------------------
# %%
# Put the embeddings into a DataFrame
#y = pd.DataFrame(msg_embeddings)
y = torch.tensor(msg_embeddings, dtype = torch.get_default_dtype())

# Prior over X
k = 2 # Number of dimensions for latent space
X_prior_mean = torch.zeros(y.size(1), k)

# Kernel definition
kernel = gp.kernels.RBF(input_dim = k, lengthscale = torch.ones(2))

# Clone the prior mean so it doesn't change during training
X = Parameter(X_prior_mean.clone())

Xu = stats.resample(X_prior_mean.clone(), 32)
gplvm = gp.models.SparseGPRegression(X, y, kernel, Xu, 
                                     noise=torch.tensor(0.01), 
                                     jitter = 1e-5)

gplvm.X = pyro.nn.PyroSample(dist.Normal(X_prior_mean, 0.1).to_event())
gplvm.autoguide("X", dist.Normal)


# %%
losses = gp.util.train(gplvm, num_steps = 4000)
plt.plot(losses)
plt.show()

# %%
gplvm.mode = "guide"
X = gplvm.X


# %%
labels = [i[0] for i in msgID_list]
X = gplvm.X_loc.detach().numpy()
# %%

pd.DataFrame(X).to_csv("twoDimNoSup.csv")


# %%
pd.DataFrame(msg_embeddings).to_csv("msgEmbeddings.csv")

# %%
X_new = pd.DataFrame(gplvm.X_loc.detach().numpy())
model.fit(X_new.iloc[:,0].tolist(), embedding)


# %%
