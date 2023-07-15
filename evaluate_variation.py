import glob

import tensorflow as tf
import numpy as np

import metrics

model_list = [# "fc_mondi_1_regression.h5", "fc_yolov7_2_regression.h5",
              # "cnn_mondi_1_regression.h5", "cnn_yolov7_1_regression.h5",
              "auto_mondi_1_regression.h5", "auto_yolov7_1_regression.h5",
             ]

rmse_metrics = tf.keras.metrics.RootMeanSquaredError()

with open(f"eval_metrics.txt", "w") as eval_f:
    eval_f.write("")

for model_name in model_list:
    
    model = tf.keras.models.load_model(model_name, compile=False)
    model.compile(optimizer='adam', loss = 'mse', metrics=['mae', rmse_metrics, metrics.R_squared])
    print(model_name)
    for defect_class in ["A", "B", "C", "D", "E"]:
        arr = np.load(f"np_data/{defect_class}_arr.npy")
        print(f"arr shape: {arr.shape}")
        y = np.expand_dims(np.load(f"np_data/{defect_class}_angle.npy"), axis=-1)
        print(f"y shape: {y.shape}")
        eval_metrics = model.evaluate(arr, y)
        with open(f"eval_metrics.txt", "a") as eval_f:
            eval_f.write(f"{defect_class}: {eval_metrics}\n")
    
    for current_class in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        arr = np.load(f"np_data/current_{current_class}_arr.npy")
        print(f"arr shape: {arr.shape}")
        y = np.expand_dims(np.load(f"np_data/current_{current_class}_angle.npy"), axis=-1)
        print(f"y shape: {y.shape}")
        eval_metrics = model.evaluate(arr, y)
        with open(f"eval_metrics.txt", "a") as eval_f:
            eval_f.write(f"{current_class}: {eval_metrics}\n")
    # print(y_predict.shape)
    # print(mse(y, y_predict))