import tensorflow as tf

model = tf.keras.models.load_model('linear.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("linear.tflite", "wb").write(tflite_model)

