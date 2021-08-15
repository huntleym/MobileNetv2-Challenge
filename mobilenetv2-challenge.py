import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

save_model_name = '/model/mobilenetv2-challenge.h5' #/model/

#initialize input size of 224x224
IMG_SIZE = (224, 224)

IMG_SHAPE = IMG_SIZE + (3,) #add rgb layer
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')

#freeze convolutional base
base_model.trainable = False

#view model
base_model.summary()

#use GlobalAveragePooling2D to convert features to single xxxx-element vector
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

#logistic layer
prediction_layer = tf.keras.layers.Dense(3000, activation='softmax', kernel_regularizer='l2')

# chain together model layers
x = base_model.output
x = global_average_layer(x)
outputs = prediction_layer(x)
model = tf.keras.Model(base_model.input, outputs)

#compile model - fine-tuning optional
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

model.summary()

# save model and architecture to file
model.save(save_model_name)
