import numpy as np 
import pandas as pd 

fullLibraryPath = '/Users/Nikki/Dropbox/UNC/Causal NLP/Reback_TxtLibrary/Reback_Project Tech Support Text Message Library_NF.xlsx'
fullLibrary = pd.read_excel(fullLibraryPath, sheet_name = 'Library', skiprows = 24, names=['txtID', 'txt'])




