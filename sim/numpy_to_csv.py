# import necessary libraries
import pandas as pd
import numpy as np


with open('data.npy', 'rb') as f:
    a = np.load(f, allow_pickle=True)

df = pd.DataFrame(a)
  
df.to_csv("data.csv")

# # load csv module
# import csv

# # open file for reading
# with open('file.csv') as csvDataFile:

#     # read file as csv file 
#     csvReader = csv.reader(csvDataFile)

#     # for every row, print the row
#     for row in csvReader:
#         print(row)