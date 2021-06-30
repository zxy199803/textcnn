import datetime
import os

class Config:
    def __init__(self):
        self.train_path = 'data/Dataset/train.txt'  # 训练集地址
        self.validation_path = 'data/Dataset/validation.txt'  # 验证集地址
        self.text_path = 'data/Dataset/test.txt'  # 测试集地址
        self.word2vec_path = 'data/Dataset/wiki_word2vec_50.bin'  # 训练好的word2vec地址
        self.word2id_path = 'data/temp/word2id.txt'  # 词id表地址
        self.word_vecs_path = "data/temp/word_vecs.npy"  # 数据集中词的词向量地址
        self.max_sen_len = 75  # 句子最大长度
        self.vec_len = 50  # 词向量长度


class CNNConfig:
    def __init__(self):
        self.kernel_size = [3, 5, 7]  # 滑动卷积窗口大小
        self.class_num = 2
        self.kernel_regularizer = None
        self.optimizer = 'sgd'
        self.dropout = 0.1

        self.batch_size = 64
        self.epochs = 40


        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'kernel_size357'
        logdir = os.path.join('logs', current_time)
        self.log_dir = logdir


class BiLSTMconfig:
    def __init__(self):
        self.class_num = 2
        self.units = 128  # 隐藏层维度
        self.dropout = 0.5
        self.optimizer = 'adam'

        self.batch_size = 128
        self.epochs = 5
