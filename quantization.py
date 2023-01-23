import tensorflow as tf

model = tf.keras.models.load_model("/home/rahul/Downloads/NLP/lstm/lstm.h5")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.post_training_quantize = True
converter.allow_custom_ops = True

converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

open("model.tflite",'wb').write(tflite_model)