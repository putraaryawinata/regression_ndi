from tensorflow.keras import callbacks as cb
import os

class AvoidOverfitting(cb.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_mae') - logs.get('mae') > 10:
            print("\nAvoid overfitting condition")
            self.model.stop_training == True

avoid_overfitting = AvoidOverfitting()
def best_ckpt(filename, dir='savedmodel', monitor='val_R_squared', save_best_only=True):
    filepath = os.path.join(dir, filename)
    saved_best_ckpt = cb.ModelCheckpoint(filepath, monitor=monitor, save_best_only=save_best_only)
    return saved_best_ckpt
early_stopping = cb.EarlyStopping(monitor='val_loss', patience=10)