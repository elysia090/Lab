import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# クリフォード代数を用いた畳み込み層の定義
class CliffordConvolution(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size, strides=1):
        super(CliffordConvolution, self).__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides
        self.conv_layer = tf.keras.layers.Conv2D(num_filters, filter_size, strides=strides, padding='SAME')
        self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv_layer(inputs)
        x = self.batchnorm(x)
        return x

# Leaky ReLUを用いた活性化関数の定義
def leaky_relu(inputs):
    return tf.nn.leaky_relu(inputs, alpha=0.1)

# クリフォード代数を用いた畳み込みニューラルネットワークのモデル定義
class CliffordCNN(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(CliffordCNN, self).__init__()
        self.conv1 = CliffordConvolution(num_filters=32, filter_size=3)
        self.activation1 = tf.keras.layers.Activation(leaky_relu)
        self.pooling1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.conv2 = CliffordConvolution(num_filters=64, filter_size=3)
        self.activation2 = tf.keras.layers.Activation(leaky_relu)
        self.pooling2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.conv3 = CliffordConvolution(num_filters=128, filter_size=3)
        self.activation3 = tf.keras.layers.Activation(leaky_relu)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.fc1 = tf.keras.layers.Dense(512, activation=leaky_relu)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.activation1(x)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pooling2(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# MNISTデータセットの読み込みと前処理
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# データ拡張
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False
)
datagen.fit(train_images)

# モデルの構築とコンパイル
model = CliffordCNN(num_classes=10)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# モデルのトレーニング
model.fit(datagen.flow(train_images, train_labels, batch_size=64), epochs=15, validation_data=(test_images, test_labels))

# テストデータでの評価
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy}")
