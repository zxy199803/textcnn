import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, GlobalAvgPool1D, Dense, Concatenate,Dropout


class TextCNN(Model):
    def __init__(self, cnn_config):
        super(TextCNN, self).__init__()
        self.kernel_sizes = cnn_config.kernel_size

        self.conv1s = []
        self.avgpools = []
        for kernel_size in cnn_config.kernel_size:
            self.conv1s.append(Conv1D(filters=128, kernel_size=kernel_size, activation='relu',
                                      kernel_regularizer=cnn_config.kernel_regularizer))
            self.avgpools.append(GlobalAvgPool1D())
        self.dropout = Dropout(cnn_config.dropout)
        self.classifier = Dense(cnn_config.class_num, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        conv1s = []
        for i in range(len(self.kernel_sizes)):
            c = self.conv1s[i](inputs)
            c = self.avgpools[i](c)
            conv1s.append(c)
        x = Concatenate()(conv1s)
        x = self.dropout(x)
        output = self.classifier(x)
        return output
