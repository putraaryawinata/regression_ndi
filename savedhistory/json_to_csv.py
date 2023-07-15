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

namefiles = ['fc_yolov7_2',] #'auto_yolov7_1', 'cnn_mondi_1',
             #'cnn_yolov7_1', 'fc_mondi_1', 'fc_yolov7_1',]

for namef in namefiles:
    jsontocsv(namef)
# for metric in metrics:
#     print(type(data[metric]))
#     break
