import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 5)

import json
 
# Opening JSON file
with open('savedhistory/autoencoder.json') as json_file:
    autoencoder = json.load(json_file)
with open('savedhistory/cnn.json') as json_file:
    cnn = json.load(json_file)
with open('savedhistory/fc.json') as json_file:
    fc = json.load(json_file)

def plotting(metrics, path_dir='savedhistory', hist_dict=[autoencoder, cnn, fc]):
    plt.plot(autoencoder[metrics], label='autoencoder')
    plt.plot(cnn[metrics], label='cnn')
    plt.plot(fc[metrics], label='fc')
    plt.legend()
    plt.savefig(f'{path_dir}/{metrics}.png', dpi=300)
    plt.close()

metrics_list = ['root_mean_squared_error', 'mae', 'R_squared', 'val_root_mean_squared_error', 'val_mae', 'val_R_squared']
for metric in metrics_list:
    plotting(metric)
    print(f"{metric} have been plotted!")
