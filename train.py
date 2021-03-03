import h5py
import numpy as np

from models.train_model import train_model
from keras.callbacks import ModelCheckpoint


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


def train():
    srcnn_model = train_model()
    print(srcnn_model.summary())
    data, label = read_training_data("dataset/train.h5")
    val_data, val_label = read_training_data("dataset/test.h5")

    checkpoint = ModelCheckpoint("weights/new_weights.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    srcnn_model.fit(data, label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, nb_epoch=200, verbose=1)


if __name__ == "__main__":
    train()