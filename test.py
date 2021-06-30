from kashgari.embeddings import BertEmbedding
from kashgari.tasks.classification import CNN_Model
import config
import matplotlib.pyplot as plt

my_config = config.Config()

def load_text(dataset='train'):
    data_path = {'val': my_config.validation_path, 'train': my_config.train_path,
                 'test': my_config.text_path}  # 数据集地址
    x = []
    y = []
    with open(data_path[dataset], encoding='utf8') as f:
        for line in f.readlines():
            sp = line.replace(" ","").split()
            # ['1', '服装很漂亮场景很大气演员演得也不错特技效果也非常精彩魔幻味够还是爱情主题既苍白又烂俗有人评论说表达佛家故事说实话如果不是有人这样说我真看不出来期待陈嘉上导演拍魔幻片不知道可不可以请他拍如果可以真是美事']

            if len(sp)>1:
                x.append(list(sp[1]))
                y.append(str(sp[0]))
    return x,y




x_train,y_train = load_text(dataset='train')
x_val, y_val = load_text(dataset='val')
x_test, y_test= load_text(dataset='test')




bert_embed = BertEmbedding('D:\研一课程\文本数据挖掘、知识图谱大作业\code\ChineseNERwithkashgari\chinese_L-12_H-768_A-12')
model = CNN_Model(bert_embed,sequence_length=100)
model.fit(x_train,y_train,x_val, y_val,epochs=3)
model.evaluate(x_test, y_test)