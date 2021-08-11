import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

save_model_name = '/model/mobilenetv2-challenge.h5'

#initialize input size of 224x224
IMG_SIZE = (224, 224)

IMG_SHAPE = IMG_SIZE + (3,) #add rgb layer
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')

#freeze convolutional base
base_model.trainable = False

#view model
#base_model.summary()

#add operation or two to out_relu layer to get appropriate sized vector

#use GlobalAveragePooling2D to convert features to single xxxx-element vector
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

#todo - apply tf.nn.l2_loss to last layer
#convert features to single prediction per data input
prediction_layer = tf.keras.layers.Dense(1, activation='softmax'))

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# model = tf.keras.Sequential([
#   base_model,
#   global_average_layer,
#   prediction_layer
# ])

#L2 loss
# Original loss function (ex: classification using cross entropy)
# unregularized_loss = tf.nn.sigmoid_cross_entropy_with_logits(predictions, labels)
# l2_loss = l2_regularization_penalty * (tf.nn.l2_loss(W_conv1) +
#                                        tf.nn.l2_loss(W_conv2) +
#                                        tf.nn.l2_loss(W_fc1))
# loss = tf.add(unregularized_loss, l2_loss, name='loss')

#compile model with L2 normalization
base_learning_rate = 0.0001 #GradientDescentOptimizer(0.5)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# save model and architecture to file
model.save(save_model_name)
