import config
import tensorflow.keras as keras
my_config = config.Config()


def build_word2id(saved_path=None):
    """
    构建词汇-id字典的txt文件
    :param saved_path: 词汇-id表储存地址
    :return: None
    """
    word2id = {'_PAD_': 0}
    data_path = [my_config.validation_path, my_config.train_path]  # 数据集地址
    for _path in data_path:
        with open(_path, encoding='utf8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                # ['0', '一种', '浪漫', '能', '让', '美女', '感动', '两种', '浪漫', '却', '能', '让', '美女']
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)

    with open(saved_path, 'w', encoding='utf8') as f:  # w没有文件就新建一个，如果有就清空再写入新内容
        f.write(str(word2id))


def read_word2id(path=my_config.word2id_path):
    """
    读取或创建word2id
    :param path:word2id_path
    :return: woed2id字典
    """
    try:
        with open(path, 'r', encoding='utf8') as f:  # 读入word2id字典
            word2id = eval(f.read())
    except FileNotFoundError:
        print('build word2id')
        build_word2id(path)

    return word2id


def build_word2vec(word2vec_path=my_config.word2vec_path, word2id_path=my_config.word2id_path,
                   save_path=None):
    """
    从训练好的词向量中构建词表中词的word2vec词表示
    :param word2vec_path: 训练好的word2vec地址
    :param word2id_path: word2id地址
    :param save_path: 储存word向量表示的地址，npy文件
    :return: word_vecs,numpy.array
    """

    import gensim
    import numpy as np

    # 读取或创建word2id
    word2id = read_word2id(my_config.word2id_path)
    word_num = len(word2id)  # 词数目

    # 从已有的word2vec中读取word的向量表示并写入array
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    print('vector size:', model.vector_size)
    word_vecs = np.array(np.random.uniform(-1., 1., [word_num, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]  # 第'id'个词的word2vec向量
        except KeyError:
            pass

    # 保存word_vecs
    if save_path:
        with open(save_path, 'wb') as f:
            np.save(f, word_vecs)  # npy只能保存一个numpy数组

    return word_vecs


def load_data(dataset='train'):
    import tensorflow.keras.preprocessing.sequence as sequence
    import numpy as np
    data_path = {'val': my_config.validation_path, 'train': my_config.train_path,
                 'test': my_config.text_path}  # 数据集地址

    labels = []
    texts2ids = []
    word2id = read_word2id(my_config.word2id_path)
    with open(data_path[dataset], encoding='utf8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            # ['0', '一种', '浪漫', '能', '让', '美女', '感动', '两种', '浪漫', '却', '能', '让', '美女']
            if len(sp) > 0:  # 过滤空行
                labels.append(int(sp[0]))
                word_id = []
                for word in sp[1:]:
                    try:
                        word_id.append(word2id[word])
                    except KeyError:  # 训练集和验证集未出现的词用0填充
                        word_id.append(0)
                texts2ids.append(word_id)
    texts2ids = sequence.pad_sequences(texts2ids, padding='post', truncating='post', value=0,
                                       maxlen=my_config.max_sen_len)  # 将序列整形到指定长度

    word_vecs = np.load(my_config.word_vecs_path)
    text2vecs = np.zeros((len(labels), my_config.max_sen_len, my_config.vec_len))
    for i, sentence in enumerate(texts2ids):
        for j, word_id in enumerate(sentence):
            text2vecs[i, j] = word_vecs[word_id]


    np_labels = np.array(labels)

    return text2vecs,np_labels

def plt_confusion_matrix(test_confussion):
    """可视化混淆矩阵"""
    import seaborn as sns
    import matplotlib.pyplot as plt
    label_txt = ['0','1']
    fig, ax = plt.subplots()
    sns.heatmap(test_confussion, ax=ax, cmap="Blues", vmax=10, cbar=False, annot=True, fmt="d")
    ax.set_xticklabels(label_txt, rotation=0, horizontalalignment='left', family='Times New Roman', fontsize=10)
    ax.set_yticklabels(label_txt, rotation=0, family='Times New Roman', fontsize=10)
    ax.xaxis.set_ticks_position("top")
    plt.show()

if __name__ == '__main__':
    import tensorflow as tf

    # build_word2id(my_config.word2id_path)
    # build_word2vec(save_path="data/temp/word_vecs.npy")
    a,b = load_data('test')
    print(b.shape)
    print(a.shape)

