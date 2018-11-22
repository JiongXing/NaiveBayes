# 朴素贝叶斯

朴素贝叶斯法(Naive Bayes)是基于贝叶斯定理与特征条件独立假设的分类方法。
工作原理是在已知样本特征的条件下，求出在各类别中发生的概率，然后取概率最大者为类别判断结果。

Naive Bayes 没有超参数可调，不能通过调参的办法提高准确率，只能依赖训练集样本的质量。
Naive Bayes 常用于文本分类中，比如垃圾邮件过滤。

我们用 sklearn 的内置 dataset 抓取20组新闻文章数据，大概有一万多篇文章。
我们的任务是训练出一个朴素贝叶斯模型，使其能对文章进行分类。

# 获取数据
```
from sklearn import datasets

# 抓取20组新闻文章
news = datasets.fetch_20newsgroups(data_home=".")
print("文章数量：%d" % len(news.data))
```

输出：
```
文章数量：11314
```

可以看一下获取到的数据是怎样的，我们取第一篇文章看看 ：
```
print(news.data[0])
```
输出：
```
From: lerxst@wam.umd.edu (where's my thing)
Subject: WHAT car is this!?
Nntp-Posting-Host: rac3.wam.umd.edu
Organization: University of Maryland, College Park
Lines: 15

 I was wondering if anyone out there could enlighten me on this car I saw
the other day. It was a 2-door sports car, looked to be from the late 60s/
early 70s. It was called a Bricklin. The doors were really small. In addition,
the front bumper was separate from the rest of the body. This is 
all I know. If anyone can tellme a model name, engine specs, years
of production, where this car is made, history, or whatever info you
have on this funky looking car, please e-mail.

Thanks,
- IL
   ---- brought to you by your neighborhood Lerxst ----
```

# 目标类别 
我们看下数据集中有哪些目标类别：
```
import pandas as pd

print(pd.Series(news.target_names))
```

输出：
```
0                  alt.atheism
1                comp.graphics
2      comp.os.ms-windows.misc
3     comp.sys.ibm.pc.hardware
4        comp.sys.mac.hardware
5               comp.windows.x
6                 misc.forsale
7                    rec.autos
8              rec.motorcycles
9           rec.sport.baseball
10            rec.sport.hockey
11                   sci.crypt
12             sci.electronics
13                     sci.med
14                   sci.space
15      soc.religion.christian
16          talk.politics.guns
17       talk.politics.mideast
18          talk.politics.misc
19          talk.religion.misc
```

共有19个类别。

# 划分数据集
把获取到的数据集划分为训练集和测试集，75%用作训练，25%用作测试：
```
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = \
    train_test_split(news.data, news.target, test_size=0.25)
```

# 文本特征抽取

接下来对文本数据进行特征抽取，使用 TF-IDF 方法抽取文本中的单词，并得到 onehot 编码数据。
```
# 训练集特征抽取
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
start_time = time.time()
features_train = tfidf.fit_transform(data_train)
print("耗时：{}秒".format(time.time() - start_time))
print("特征矩阵：{}".format(features_train.shape))

# 测试集特征抽取
start_time = time.time()
features_test = tfidf.transform(data_test)
print("耗时：{}秒".format(time.time() - start_time))
print("特征矩阵：{}".format(features_test.shape))
```

输出：
```
****************************** 训练集特征抽取 ******************************
耗时：2.4131357669830322秒
特征矩阵：(8485, 112394)
****************************** 测试集特征抽取 ******************************
耗时：0.728867769241333秒
特征矩阵：(2829, 112394)
```

在训练集中共抽取出112394个特征。

# 朴素贝叶斯建模
接下来用训练数据集建模，这里用 sklearn 提供的**多项式型朴素贝叶斯分类器**：
```
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
start_time = time.time()
nb.fit(features_train, target_train)
print("耗时：{}秒".format(time.time() - start_time))
```

输出：
```
****************************** 朴素贝叶斯建模 ******************************
耗时：0.09011292457580566秒
```

# 模型评估
对训练得到的模型做准确率评估：
```
print(nb.score(features_test, target_test))
```

输出：
```
****************************** 评估准确率 ******************************
0.8218451749734889
```

准确率是82.18%.

# 使用模型进行预测
用我们的模型对测试集数据分类，同时与测试集的真实分类进行对比：
```
print("真实值：")
print(target_test)
print("预测值：")
print(nb.predict(features_test))
```

输出：
```
****************************** 预测 ******************************
真实值：
[13 14 17 ... 10  0 15]
预测值：
[13 16 17 ... 10 15 15]
```

在这个数据集中，我们的模型基本能达到80%以上的分类准确率。