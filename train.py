from Model.TextCNN import TextCNN
import config
import tensorflow as tf
from utils import load_data
from sklearn import metrics


class ModelHelper:
    def __init__(self, model_config):
        self.creat_model(model_config=model_config)

    def creat_model(self, model_config):
        model = TextCNN(cnn_config=model_config)
        model.compile(optimizer=model_config.optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        self.model = model
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=model_config.log_dir, histogram_freq=1, )

    def fit(self, model_config, x_train, y_train, x_val, y_val):
        print('Train...')
        self.model.fit(x_train, y_train,
                       batch_size=model_config.batch_size,
                       epochs=model_config.epochs,
                       verbose=1,
                       validation_data=(x_val, y_val),
                       callbacks=[self.tensorboard_callback]
                       )
        self.model.summary()

    def predict(self, x_test):
        y_pred = self.model.predict(x_test, batch_size=1)
        result = []
        for y in y_pred:
            result.append(tf.argmax(y, 0).numpy())
        return result


my_model_config = config.CNNConfig()

model_helper = ModelHelper(model_config=my_model_config)

with tf.device('/gpu:0'):
    x_train, y_train = load_data('train')
    x_val, y_val = load_data('val')
    x_test, y_test = load_data('test')
    print(x_train.shape)
    model_helper.fit(my_model_config, x_train, y_train, x_val, y_val)
    loss, acc = model_helper.model.evaluate(x_test, y_test, verbose=1)
    print("loss:", loss, " acc:", acc)

    result = model_helper.predict(x_test)
    report = metrics.classification_report(y_test, result, target_names=['0', '1'], digits=4)
    print(report)
