######################### 一、小规模数据集测试  ################################################
# 导入sklearn的自带数据集，常规前处理，介绍略
import xgboost as xgb
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error as mse
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

# 读取数据
X = load_boston().data
y = load_boston().target

# 参数设置
N = 20
seed=random.randint(0,100000)

# 基线模型表现 - 一次性训练结果

# 数据集分割
X_train, X_test, y_train, y_test = ttsplit(X, y, test_size=0.1, random_state=seed)  # 分割测试集及训练集
print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (X_train.shape[0], X_test.shape[0]))

xg_train = xgb.DMatrix(X_train, label=y_train)   # 用于一次性训练的数据集
xg_test = xgb.DMatrix(X_test, label=y_test)  # 用于最终验证的数据集

# ================= 一次性训练 =====================#
params = {'objective': 'reg:squarederror'}
model_1 = xgb.train(params, xg_train, 30)    # 注意数据集指向
model_1.save_model('model_1.model')

# 效果检验，注意要使用相同的测试集
benchmark = mse(model_1.predict(xg_test), y_test)
print("The mse benchmark is %.2f" %benchmark)     # benchmark

# 训练数据集样本数目：455, 测试数据集样本数目：51 The mse benchmark is 13.19
# 为了比较一下，将数据集切分成不同数量N，模型效果的变化情况，写了如下的函数
### 函数-

def Check_Continuation(N, X_train, y_train, xg_test, y_test):
    for i in range(1,N+1):  # 将数据集切成N份，记录不同N下模型的效果
    
        if i == N:
            xg_train_1 = xgb.DMatrix(X_train_2, label=y_train_2)  # 使用最后一批剩余数据
        else:
            X_train_1, X_train_2, y_train_1, y_train_2 = ttsplit(X_train, y_train, test_size=1/N, random_state=seed)  
            # 将训练集再分割,逐批提取数据并作训练
            xg_train_1 = xgb.DMatrix(X_train_1, label=y_train_1)  # 分批提取训练数据

        if i == 1:
            model_2_v1 = xgb.train(params, xg_train_1, 30)
            model_2_v1.save_model('model_2.model')
            result = mse(model_2_v1.predict(xg_test), y_test)

        else:
            model_2_v2 = xgb.train(params, xg_train_1, 30, xgb_model='model_2.model')
            model_2_v2.save_model('model_2.model')
            result = mse(model_2_v2.predict(xg_test), y_test)

        #print("Round %d - The mse is %.2f" %(i,result))  # 
    
    return result

# 调用函数看效果，上边已经把N设成了20
import numpy as np

MSE_result = np.linspace(0,0,N)

for i in range(2,N):
    MSE_result[i-2] = Check_Continuation(i, X_train, y_train, xg_test, y_test)

print(MSE_result)

plt.figure(figsize=(12,6), facecolor='w')
ln_x_valid = range(2,N)
plt.show()
plt.plot(ln_x_valid, MSE_result[:(N-2)], 'r-', lw=2, label=u'实际值')
plt.show()
# 这个效果明细没有规律....哦，训练数据只有几百，肯定不行啊，哈哈哈哈！换个数据集~

