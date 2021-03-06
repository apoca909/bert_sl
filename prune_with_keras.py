# -*- coding: utf-8 -*-
##https://github.com/Tony607/prune-keras
import tensorflow as tf

import tempfile
import zipfile
import os

"""## Prepare the training data"""

batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

if tf.keras.backend.image_data_format() == 'channels_first':
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

"""## Train a MNIST model without pruning

### Build the MNIST model
"""

l = tf.keras.layers

model = tf.keras.Sequential([
    l.Conv2D(
        32, 5, padding='same', activation='relu', input_shape=input_shape),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.BatchNormalization(),
    l.Conv2D(64, 5, padding='same', activation='relu'),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.Flatten(),
    l.Dense(1024, activation='relu'),
    l.Dropout(0.4),
    l.Dense(num_classes, activation='softmax')
])

model.summary()

"""### Train the model to reach an accuracy >99%

Load [TensorBoard](https://www.tensorflow.org/tensorboard) to monitor the training process
"""
prefix='E:\\workspace\\keras_prune\\origin\\'
logdir = tempfile.mkdtemp(prefix=prefix, dir='melog')
print('Writing orgin training logs to ' + logdir)

# %tensorboard --logdir={logdir}

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)]

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""### Save the original model for size comparison later"""

# Backend agnostic way to save/restore models
_, keras_file = tempfile.mkstemp('.h5', prefix=prefix)
print('Saving origin model to: ', keras_file)
tf.keras.models.save_model(model, keras_file, include_optimizer=False)

"""## Train a pruned MNIST

We provide a `prune_low_magnitude()` API to train models with removed connections. The Keras-based API can be applied at the level of individual layers, or the entire model. We will show you the usage of both in the following sections.

At a high level, the technique works by iteratively removing (i.e. zeroing out) connections between layers, given an schedule and a target sparsity.

For example, a typical configuration will target a 75% sparsity, by pruning connections every 100 steps (aka epochs), starting from step 2,000. For more details on the possible configurations, please refer to the github documentation.

### Build a pruned model layer by layer
In this example, we show how to use the API at the level of layers, and build a pruned MNIST solver model.

In this case, the `prune_low_magnitude(`) 
receives as parameter the Keras layer whose weights we want pruned.

This function requires a pruning params which configures the pruning algorithm during training. Please refer to our github page for detailed documentation. The parameter used here means:


1.   **Sparsity.** PolynomialDecay is used across the whole training process. We start at the sparsity level 50% and gradually train the model to reach 90% sparsity. X% sparsity means that X% of the weight tensor is going to be pruned away.
2.   **Schedule**. Connections are pruned starting from step 2000 to the end of training, and runs every 100 steps. The reasoning behind this is that we want to train the model without pruning for a few epochs to reach a certain accuracy, to aid convergence. Furthermore, we give the model some time to recover after each pruning step, so pruning does not happen on every step. We set the pruning frequency to 100.
"""

from tensorflow_model_optimization.sparsity import keras as sparsity

"""To demonstrate how to save and restore a pruned keras model, in the following example we first train the model for 10 epochs, save it to disk, and finally restore and continue training for 2 epochs. With gradual sparsity, four important parameters are begin_sparsity, final_sparsity, begin_step and end_step. The first three are straight forward. Let's calculate the end step given the number of train example, batch size, and the total epochs to train."""

import numpy as np

prefix='E:\\workspace\\keras_prune\\prune\\'
num_train_samples = x_train.shape[0]
end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
print('End step: ' + str(end_step))

pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=2000,
                                                   end_step=end_step,
                                                   frequency=100)
}

pruned_model = tf.keras.Sequential([
    sparsity.prune_low_magnitude(
        l.Conv2D(32, 5, padding='same', activation='relu'),
        input_shape=input_shape,
        **pruning_params),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.BatchNormalization(),
    sparsity.prune_low_magnitude(l.Conv2D(64, 5, padding='same', activation='relu'), **pruning_params),
    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
    l.Flatten(),
    sparsity.prune_low_magnitude(l.Dense(1024, activation='relu'), **pruning_params),
    l.Dropout(0.4),
    sparsity.prune_low_magnitude(l.Dense(num_classes, activation='softmax'),
                                 **pruning_params)
])

pruned_model.summary()

"""Load Tensorboard"""


logdir = tempfile.mkdtemp(prefix=prefix)
print('Writing prune training logs to ' + logdir)

# %tensorboard --logdir={logdir}

"""### Train the model

Start pruning from step 2000 when accuracy >98%
"""

pruned_model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

# Add a pruning step callback to peg the pruning step to the optimizer's
# step. Also add a callback to add pruning summaries to tensorboard
callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
]

pruned_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_test, y_test))

score = pruned_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""### Save and restore the pruned model

Continue training for two epochs:
"""

_, checkpoint_file = tempfile.mkstemp('.h5', prefix=prefix)
print('Saving pruned model to: ', checkpoint_file)
# saved_model() sets include_optimizer to True by default. Spelling it out here
# to highlight.
tf.keras.models.save_model(pruned_model, checkpoint_file, include_optimizer=True)

with sparsity.prune_scope():
  restored_model = tf.keras.models.load_model(checkpoint_file)

restored_model.fit(x_train, y_train,
                   batch_size=batch_size,
                   epochs=2,
                   verbose=1,
                   callbacks=callbacks,
                   validation_data=(x_test, y_test))

score = restored_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""In the example above, a few things to note are:


*   When saving the model, include_optimizer must be set to True. We need to preserve the state of the optimizer across training sessions for pruning to work properly.
*   When loading the pruned model, you need the prune_scope() for deseriazliation.

### Strip the pruning wrappers from the pruned model before export for serving
Before exporting a serving model, you'd need to call the `strip_pruning` API to strip the pruning wrappers from the model, as it's only needed for training.
"""

final_model = sparsity.strip_pruning(pruned_model)
final_model.summary()

_, pruned_keras_file = tempfile.mkstemp('.h5', prefix=prefix)
print('Saving pruned model to: ', pruned_keras_file)

# No need to save the optimizer with the graph for serving.
tf.keras.models.save_model(final_model, pruned_keras_file, include_optimizer=False)

"""### Compare the size of the unpruned vs. pruned model after compression"""

_, zip1 = tempfile.mkstemp('.zip', prefix=prefix)
with zipfile.ZipFile(zip1, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(keras_file)
print("Size of the unpruned model before compression: %.2f Mb" % (os.path.getsize(keras_file) / float(2**20)))
print("Size of the unpruned model after compression: %.2f Mb" % (os.path.getsize(zip1) / float(2**20)))

_, zip2 = tempfile.mkstemp('.zip', prefix=prefix)
with zipfile.ZipFile(zip2, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(pruned_keras_file)
print("Size of the pruned model before compression: %.2f Mb" %  (os.path.getsize(pruned_keras_file) / float(2**20)))
print("Size of the pruned model after compression: %.2f Mb" %  (os.path.getsize(zip2) / float(2**20)))

"""### Prune a whole model

The `prune_low_magnitude` function can also be applied to the entire Keras model. 

In this case, the algorithm will be applied to all layers that are ameanable to weight pruning (that the API knows about). Layers that the API knows are not ameanable to weight pruning will be ignored, and unknown layers to the API will cause an error.

*If your model has layers that the API does not know how to prune their weights, but are perfectly fine to leave "un-pruned", then just apply the API in a per-layer basis.*

Regarding pruning configuration, the same settings apply to all prunable layers in the model.

Also noteworthy is that pruning doesn't preserve the optimizer associated with the original model. As a result, it is necessary to re-compile the pruned model with a new optimizer.

Before we move forward with the example, lets address the common use case where you may already have a serialized pre-trained Keras model, which you would like to apply weight pruning on. We will take the original MNIST model trained previously to show how this works. In this case, you start by loading the model into memory like this:
"""

# Load the serialized model
loaded_model = tf.keras.models.load_model(keras_file)
prefix = 'E:\\workspace\\keras_prune\\pretrained\\'
"""Then you can prune the model loaded and compile the pruned model for training. In this case training will restart from step 0. Given the model we loadded already reached a satisfactory accuracy, we can start pruning immediately. As a result, we set the begin_step to 0 here, and only train for another four epochs."""

epochs = 10
end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
print(end_step)

new_pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=0,
                                                   end_step=end_step,
                                                   frequency=100)
}

new_pruned_model = sparsity.prune_low_magnitude(model, **new_pruning_params)
new_pruned_model.summary()

new_pruned_model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

"""Load tensorboard"""

logdir = tempfile.mkdtemp(prefix=prefix)
print('Writing training logs to ' + logdir)

# %tensorboard --logdir={logdir}

"""### Train the model for another four epochs"""

# Add a pruning step callback to peg the pruning step to the optimizer's
# step. Also add a callback to add pruning summaries to tensorboard
callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
]

new_pruned_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_test, y_test))

score = new_pruned_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""### Export the pruned model for serving"""

final_model = sparsity.strip_pruning(pruned_model)
final_model.summary()

_, new_pruned_keras_file = tempfile.mkstemp('.h5', prefix=prefix)
print('Saving pretrain pruned model to: ', new_pruned_keras_file)
tf.keras.models.save_model(final_model, new_pruned_keras_file,
                        include_optimizer=False)

"""The model size after compression is the same as the one pruned layer-by-layer"""

_, zip3 = tempfile.mkstemp('.zip', prefix=prefix)
with zipfile.ZipFile(zip3, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(new_pruned_keras_file)
print("Size of the pruned model before compression: %.2f Mb"% (os.path.getsize(new_pruned_keras_file) / float(2**20)))
print("Size of the pruned model after compression: %.2f Mb" % (os.path.getsize(zip3) / float(2**20)))

"""## Convert to TensorFlow Lite

Finally, you can convert the pruned model to a format that's runnable on your targeting backend. Tensorflow Lite is an example format you can use to deploy to mobile devices. To convert to a Tensorflow Lite graph, you need to use the TFLiteConverter as below:

### Convert the model with TFLiteConverter
"""

tflite_model_file = prefix + '/sparse_mnist.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model_file(pruned_keras_file)
tflite_model = converter.convert()
with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)

"""### Size of the TensorFlow Lite model after compression"""

_, zip_tflite = tempfile.mkstemp('.zip', prefix=prefix)
with zipfile.ZipFile(zip_tflite, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(tflite_model_file)
print("Size of the tflite model before compression: %.2f Mb"% (os.path.getsize(tflite_model_file) / float(2**20)))
print("Size of the tflite model after compression: %.2f Mb" % (os.path.getsize(zip_tflite) / float(2**20)))

"""### Evaluate the accuracy of the TensorFlow Lite model"""

import numpy as np

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def eval_model(interpreter, x_test, y_test):
  total_seen = 0
  num_correct = 0

  for img, label in zip(x_test, y_test):
    inp = img.reshape((1, 28, 28, 1))
    total_seen += 1
    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    if np.argmax(predictions) == np.argmax(label):
      num_correct += 1

    if total_seen % 1000 == 0:
        print("Accuracy after %i images: %f" %
              (total_seen, float(num_correct) / float(total_seen)))

  return float(num_correct) / float(total_seen)

print(eval_model(interpreter, x_test, y_test))

"""### Post-training quantize the TensorFlow Lite model

You can combine pruning with other optimization techniques like post training quantization. As a recap, post-training quantization converts weights to 8 bit precision as part of model conversion from keras model to TFLite's flat buffer, resulting in a 4x reduction in the model size.

In the following example, we take the pruned keras model, convert it with post-training quantization, check the size reduction and validate its accuracy.
"""

converter = tf.lite.TFLiteConverter.from_keras_model_file(pruned_keras_file)

converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

tflite_quant_model = converter.convert()

tflite_quant_model_file = '/tmp/sparse_mnist_quant.tflite'
with open(tflite_quant_model_file, 'wb') as f:
    f.write(tflite_quant_model)

_, zip_tflite = tempfile.mkstemp('.zip', prefix=prefix)
with zipfile.ZipFile(zip_tflite, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(tflite_quant_model_file)
print("Size of the tflite model before compression: %.2f Mb"% (os.path.getsize(tflite_quant_model_file) / float(2**20)))
print("Size of the tflite model after compression: %.2f Mb" % (os.path.getsize(zip_tflite) / float(2**20)))

"""The size of the quantized model is roughly 1/4 of the orignial one."""

interpreter = tf.lite.Interpreter(model_path=str(tflite_quant_model_file))
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

print(eval_model(interpreter, x_test, y_test))

"""## Conclusion

In this tutorial, we showed you how to create *sparse models* with the TensorFlow model optimization toolkit weight pruning API. Right now, this allows you to create models that take significant less space on disk. The resulting model can also be more efficiently implemented to avoid computation; in the future TensorFlow Lite will provide such capabilities.

More specifically, we walked you through an end-to-end example of training a simple MNIST model that used the weight pruning API. We showed you how to convert it to the Tensorflow Lite format for mobile deployment, and demonstrated how with simple file compression the model size was reduced 5x.

We encourage you to try this new capability on your Keras models, which can be particularly important for deployment in resource-constraint environments.
"""