import matplotlib.pyplot as plt
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
import numpy as np
import pickle
from keras import backend as K
import tensorflow.compat.v1 as tf

config =  tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

x_train = pickle.load(open('x_train.pkl','rb'))
x_valid = pickle.load(open('y_train.pkl','rb'))
y_train = pickle.load(open('x_valid.pkl','rb'))
y_valid = pickle.load(open('y_valid.pkl','rb'))

# fig,axs=plt.subplots(1,2)
# axs[0].imshow(x_valid[0].squeeze())
# axs[1].imshow(y_valid[0].squeeze())
# plt.show()


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    x=Conv2D(filters=n_filters,kernel_size=(kernel_size,kernel_size),padding='same')(input_tensor)
    if batchnorm:
        x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1=MaxPooling2D((2,2))(c1)
    p1 = Dropout(dropout*0.5)(p1)
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)


    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

input=Input((128,128,3))



train_aug= ImageDataGenerator(
    rotation_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,

)

def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
  denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))
  return 1 - numerator / denominator


def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    global precision,recall
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def mean_iou(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

def dice_single(true,pred):
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = K.round(pred)

    intersection = K.sum(true * pred, axis=-1)
    true = K.sum(true, axis=-1)
    pred = K.sum(pred, axis=-1)

    return ((2*intersection) + K.epsilon()) / (true + pred + K.epsilon())


model=unet(input,n_filters=16,dropout=0.5,batchnorm=True)

model.compile(optimizer="Adam",
              loss=dice_loss,
              metrics=['accuracy',  precision, recall,f1_m,mean_iou,dice_single])

callbacks=[EarlyStopping(patience=15,verbose=1),ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
           ModelCheckpoint("model_tumor_brain.h5",verbose=1, save_best_only=True,save_weights_only=False)]

results=model.fit(x_train,y_train,
                  batch_size=32,
                  epochs=100,
                  callbacks=callbacks,
                  validation_data=(x_valid,y_valid))


plt.figure(figsize=(8, 8))
plt.title("Learning")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="g")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()


import matplotlib.pyplot as plt
from keras.models import Model, load_model
import numpy as np
import pickle
from keras.models import Model, load_model
model=load_model("model_tumor_brain.h5", custom_objects={'dice_loss': dice_loss,'precision': precision, 'recall': recall,'f1_m': f1_m,'mean_iou': mean_iou,'dice_single': dice_single})

import cv2

test_src='C:/Users/kamil/Desktop/MLnauka/brain tumor test/2.png'
img=cv2.imread(test_src)
img=cv2.resize(img,(128,128))
img=np.asarray(img)/255.0
img = np.expand_dims(img, axis=0)

predict = model.predict(img)
print(predict)
predict = (predict >0.4).astype(np.float32)
fig, ax = plt.subplots(1,2,figsize=(12,18))
ax[1].imshow(np.squeeze(img))
ax[0].imshow(np.squeeze(img))
ax[0].contour(np.squeeze(predict[0]), vmin=0, vmax=1, colors="r")

plt.show()