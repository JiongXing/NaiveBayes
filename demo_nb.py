import time
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def print_line(title):
    print("*" * 30 + " {} ".format(title) + "*" * 30)


# 抓取20组新闻文章
news = datasets.fetch_20newsgroups(data_home=".")
print("文章数量：%d" % len(news.data))

print_line("第一篇")
print(news.data[0])

print_line("目标类别")
print(pd.Series(news.target_names))

# 划分数据集
data_train, data_test, target_train, target_test = \
    train_test_split(news.data, news.target, test_size=0.25)

# 特征抽取
print_line("训练集特征抽取")
tfidf = TfidfVectorizer()
start_time = time.time()
features_train = tfidf.fit_transform(data_train)
print("耗时：{}秒".format(time.time() - start_time))
print("特征矩阵：{}".format(features_train.shape))

print_line("测试集特征抽取")
start_time = time.time()
features_test = tfidf.transform(data_test)
print("耗时：{}秒".format(time.time() - start_time))
print("特征矩阵：{}".format(features_test.shape))

# 朴素贝叶斯
print_line("朴素贝叶斯建模")
nb = MultinomialNB()
start_time = time.time()
nb.fit(features_train, target_train)
print("耗时：{}秒".format(time.time() - start_time))

# 评估
print_line("评估准确率")
print(nb.score(features_test, target_test))

# 预测
print_line("预测")
print("真实值：")
print(target_test)
print("预测值：")
print(nb.predict(features_test))
