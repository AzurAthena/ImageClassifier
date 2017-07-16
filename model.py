from getdata import load
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K


def scheduler(epoch):
    current_lr = 0.01
    epoch_step = 2
    if epoch == 0:
        updated_lr = current_lr
    elif epoch % epoch_step == 0:
        dividend = epoch // epoch_step
        updated_lr = current_lr/dividend
    else:
        updated_lr = current_lr
    return updated_lr


lr_scheduler = LearningRateScheduler(scheduler)

EPOCHS = 3
K.set_image_dim_ordering('th')

x_train, x_test, y_train, y_test = load(test_size=0.4)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3), padding='same', input_shape=(3, 100, 100)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# Create optimizer
optimizer = Adam(lr=0.01)

# Compile and fir model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

check = ModelCheckpoint("model_checkpoints/weights.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
model.fit(x_train, y_train, batch_size=32, nb_epoch=EPOCHS, callbacks=[check, lr_scheduler], validation_data=(x_test,y_test))

out = model.predict(x_test, batch_size=32)
