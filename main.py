# 获取训练数据
import numpy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
print("获得训练数据")
#获得训练数据
train_data =pd.read_csv('train_set.csv', sep='\t', nrows=15000)

print(train_data)
print("数据处理")
#对词项用idf值进行改进
tfidfVectorizer = TfidfVectorizer()
train_data_tfidfVectorizer = tfidfVectorizer.fit_transform(train_data.text)
#从样本中随机的按比例选取train data和testdata,
print("训练模型")
X_train, X_test, y_train, y_test = train_test_split(train_data_tfidfVectorizer, train_data.label, test_size=0.25, random_state=0)
#岭回归分类器
clf = RidgeClassifier()
clf.fit(X_train, y_train)

test_prediction = clf.predict(X_test)
score_f1 = f1_score(y_test, test_prediction, average='macro')
#正确率
print('准确率: %.4f' % score_f1)
# 模型预测结果 : 0.8826
test_df = pd.read_csv('test_a.csv')
test_tfidf = tfidfVectorizer.transform(test_df.text)
test_pre = clf.predict(test_tfidf)
print(type(test_pre))
#测试数据
#获得测试结果文件csv,result06.csv,注意第一行前面要加label，否者平台无法识别
print("获得测试结果文件csv,result06.csv,注意第一行前面要加label，否者平台无法识别")
numpy.savetxt("result06.csv", test_pre, delimiter=" ",fmt="%i")
