import json
import pandas as pd

def jsontocsv(namefile):
    with open(f"{namefile}.json", "r") as jsonfile:
        data = json.load(jsonfile)

    metrics = list(data.keys())
    df = pd.DataFrame.from_dict(data, orient="index").T
    print(df.head())
    df.to_csv(f'{namefile}.csv')
    print(f"{namefile} has been created!")

namefiles = ['autoencoder_mondi', 'autoencoder', 'cnn_mondi',
             'cnn', 'fc_mondi', 'fc',]

for namef in namefiles:
    jsontocsv(namef)
# for metric in metrics:
#     print(type(data[metric]))
#     break
