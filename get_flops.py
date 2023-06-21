import tensorflow as tf
import keras.backend as K
import metrics

from keras_flops import get_flops


# def get_flops(model_h5_path):
#     session = tf.compat.v1.Session()
#     graph = tf.compat.v1.get_default_graph()


#     with graph.as_default():
#         with session.as_default():
#             model = tf.keras.models.load_model(model_h5_path, compile=False)
#             model.compile(optimizer='adam', loss='mse', metrics=['mae', rmse_metrics, r2])

#             run_meta = tf.compat.v1.RunMetadata()
#             opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

#             # Optional: save printed results to file
#             # flops_log_path = os.path.join(tempfile.gettempdir(), 'tf_flops_log.txt')
#             # opts['output'] = 'file:outfile={}'.format(flops_log_path)

#             # We use the Keras session graph in the call to the profiler.
#             flops = tf.compat.v1.profiler.profile(graph=graph,
#                                                   run_meta=run_meta, cmd='op', options=opts)

#             return flops.total_float_ops


# .... Define your model here ....
rmse_metrics = tf.keras.metrics.RootMeanSquaredError()
r2 =  metrics.R_squared

model = tf.keras.models.load_model('auto_mondi_1_regression.h5', compile=False)
model.compile(optimizer='adam',
              loss = 'mse',
              metrics=['mae', rmse_metrics, r2])
# You need to have compiled your model before calling this.
flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
# print(get_flops('fc_mondi_1_regression.h5'))