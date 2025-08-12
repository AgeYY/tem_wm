# import tensorflow as tf
# print(tf.__version__)
# try:
#     print(tf.config.list_physical_devices('GPU'))
# except Exception as e:
#     print("GPU init error:", e)

import tensorflow as tf
print("TF:", tf.__version__)
print("GPU built:", tf.test.is_built_with_cuda())
print("GPUs:", tf.config.list_physical_devices("GPU"))

# optional: avoid pre-allocating all memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("set_memory_growth failed:", e)

# 1) Matmul on GPU (hits cuBLAS)
with tf.device('/GPU:0'):
    a = tf.random.normal([4096, 4096])
    b = tf.random.normal([4096, 4096])
    c = tf.matmul(a, b)  # should run on GPU
print("Matmul OK, c shape:", c.shape)

# 2) Simple conv (hits cuDNN)
import numpy as np
with tf.device('/GPU:0'):
    x = tf.random.normal([8, 64, 64, 32])  # NHWC
    w = tf.random.normal([3, 3, 32, 64])
    y = tf.nn.conv2d(x, w, strides=1, padding='SAME')
print("Conv2D OK, y shape:", y.shape)
