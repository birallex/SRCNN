from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.optimizers import SGD, Adam

def train_model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(128, (9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(32, 32, 1)))
    SRCNN.add(Conv2D(64, (3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(1, (5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    
    return SRCNN