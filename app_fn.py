# 导入所有需要的包
from numpy.core.fromnumeric import shape
import streamlit as st
import yfinance as yf
import pandas as pd
import datetime 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# 获取数据 , 查看数据
today = datetime.date.today()

stock_name = input('请输入一个股票名字:如 AAPL,FB,SPGI.  ')

df = yf.download(stock_name,start = '2015-01-02',progress = False)
print(df.info)
st.write(df)
# 将当日股票数据设成输入数据，将次日股票数据设成目标数据
dfs_input = df[['Open','High','Low','Close','Volume','Adj Close']].to_numpy()
dfs_target = df[['Adj Close']].to_numpy()

df_input = dfs_input[0:len(dfs_target)-1] # 从2010年01月04日到第 n - 1 天的股票数据
# 可以用 print(df_input.shape) 查看矩阵大小
# 可以用 print(df_input[:]) 查看矩阵数据


df_target = dfs_target[1:len(dfs_target)] # 从2021年12月14日到第 n 天的股票数据
# 可以用 print(df_target.shape) 查看矩阵大小
# 可以用 print(df_target[:]) 查看矩阵数据

train_input, test_input, train_target, test_target = train_test_split(df_input, df_target , random_state = 42) 

# 用多项线性回归模型训练数据  y=ax1+bx2+cx3+dx4+ex5+f
lr = LinearRegression()
lr.fit(train_input, train_target)

lr.score(train_input, train_target)
lr.score(test_input, test_target)

# 可以用 print(lr.coef_, lr.intercept_) 查看加权值abcdef

# 用多项线性回归模型预测次日最终股票交易金额
a = np.dot(lr.coef_, dfs_input[-1])
b = lr.intercept_

print('根据多元回归预测的{}股票在{}的最终成交金额为{}'.format(stock_name,today,a + b))
st.write('根据多元回归预测的{}股票在{}的最终成交金额为{}'.format(stock_name,today,a + b))




## 用多元回归算法预测各个数据 y=ax**2 + bx + c

ac_dfs_target = df[['Adj Close']].to_numpy()
ac_df_target = ac_dfs_target[1:len(ac_dfs_target)]

ac_df_input = np.array(range(0,len(ac_dfs_target))) # 根据目标数量添加序号
ac_df_input = ac_df_input.reshape(len(ac_dfs_target),1)
ac_df_input = ac_df_input[0:len(ac_dfs_target)-1]

ac_train_input, ac_test_input, ac_train_target, ac_test_target = train_test_split(ac_df_input, ac_df_target , random_state = 42) 

train_poly = np.column_stack((ac_train_input**2,ac_train_input))
test_poly = np.column_stack((ac_test_input**2,ac_test_input))

lr = LinearRegression()
lr.fit(train_poly, ac_train_target)

# 可以用print(lr.coef_, lr.intercept_)查看加权值abc

print('根据多项回归预测的{}股票在{}的最终成交金额为{}'.format(stock_name,today,lr.predict([[len(ac_dfs_target)**2, len(ac_dfs_target)]])))

point = np.arange(0,len(ac_df_target))

plt.plot(ac_df_input,ac_df_target)
plt.plot(point, lr.coef_[:,0]* point**2 + lr.coef_[:,1]*point + lr.intercept_)

plt.scatter(len(ac_df_target), lr.predict([[len(ac_dfs_target)**2, len(ac_dfs_target)]]), marker='^')
plt.xlabel('date')
plt.ylabel('open')
plt.legend(['k_line','pr_line','adj_close'])
plt.show()




### 用线性回归算法预测各个数据  y=ax + b

sac_dfs_target = df[['Adj Close']].to_numpy()
sac_df_target = sac_dfs_target[1:len(sac_dfs_target)]

sac_df_input = np.array(range(0,len(sac_dfs_target))) # 根据目标数量添加序号
sac_df_input = sac_df_input.reshape(len(sac_dfs_target),1)
sac_df_input = sac_df_input[0:len(sac_dfs_target)-1]

sac_train_input, sac_test_input, sac_train_target, sac_test_target = train_test_split(sac_df_input, sac_df_target , random_state = 42) 

lr = LinearRegression()
lr.fit(sac_train_input, sac_train_target)

# 可以用print(lr.coef_, lr.intercept_)查看加权值ab

print('根据线性回归预测的{}股票在{}的最终成交金额为{}'.format(stock_name,today, lr.predict([[len(sac_dfs_target)+1]])))

plt.plot(sac_df_input,sac_df_target)

plt.plot([0,len(sac_df_input)], [lr.coef_[:,0]*0+lr.intercept_ , lr.coef_[:,0]*len(sac_df_input)+lr.intercept_],color='black')

plt.scatter(len(sac_dfs_target), lr.predict([[len(sac_dfs_target)]]), marker='^')
plt.xlabel('date')
plt.ylabel('open')
plt.legend(['k_line','lr_line','adj_close'])
plt.show()





##### 用k邻居算法预测各个数据(5日均线)

f_df_target = df[['Adj Close']].to_numpy()

f_df_input = np.array(range(0,len(f_df_target))) # 根据目标数量添加序号
f_df_input = f_df_input.reshape(len(f_df_target),1)

f_train_input, f_test_input, f_train_target, f_test_target = train_test_split(f_df_input, f_df_target , random_state = 42) 

knr = KNeighborsRegressor()
knr.fit(f_train_input, f_train_target)

knr.n_neighbors = 5
print('根据5日均线预测的{}股票在{}的最终成交金额为{}'.format(stock_name,today, knr.predict([[len(f_df_target)+1]])))



plt.plot(f_df_input , f_df_target,color='blue')
plt.plot(f_df_input , knr.predict(f_df_input),color='red')

plt.scatter(len(f_df_target) , knr.predict([[len(f_df_target)+1]]), marker='^')
plt.xlabel('Date')
plt.ylabel('Adj_close')
plt.legend(['k_line','5_line','adj_close'])
plt.show()

##### 用k邻居算法预测各个数据(10日均线)

f_df_target = df[['Adj Close']].to_numpy()

f_df_input = np.array(range(0,len(f_df_target))) # 根据目标数量添加序号
f_df_input = f_df_input.reshape(len(f_df_target),1)

f_train_input, f_test_input, f_train_target, f_test_target = train_test_split(f_df_input, f_df_target , random_state = 42) 

knr = KNeighborsRegressor()
knr.fit(f_train_input, f_train_target)

knr.n_neighbors = 10
print('根据10日均线预测的{}股票在{}的最终成交金额为{}'.format(stock_name,today, knr.predict([[len(f_df_target)+1]])))

plt.plot(f_df_input , f_df_target,color='blue')
plt.plot(f_df_input , knr.predict(f_df_input),color='red')

plt.scatter(len(f_df_target) , knr.predict([[len(f_df_target)+1]]), marker='^')
plt.xlabel('Date')
plt.ylabel('Adj_close')
plt.legend(['k_line','10_line','adj_close'])
plt.show()


##### 用k邻居算法预测各个数据(20日均线)

f_df_target = df[['Adj Close']].to_numpy()

f_df_input = np.array(range(0,len(f_df_target))) # 根据目标数量添加序号
f_df_input = f_df_input.reshape(len(f_df_target),1)

f_train_input, f_test_input, f_train_target, f_test_target = train_test_split(f_df_input, f_df_target , random_state = 42) 

knr = KNeighborsRegressor()
knr.fit(f_train_input, f_train_target)

knr.n_neighbors = 20
print('根据20日均线预测的{}股票在{}的最终成交金额为{}'.format(stock_name,today, knr.predict([[len(f_df_target)+1]])))

plt.plot(f_df_input , f_df_target,color='blue')
plt.plot(f_df_input , knr.predict(f_df_input),color='red')

plt.scatter(len(f_df_target) , knr.predict([[len(f_df_target)+1]]), marker='^')
plt.xlabel('Date')
plt.ylabel('Adj_close')
plt.legend(['k_line','20_line','adj_close'])
plt.show()


##### 用k邻居算法预测各个数据(30日均线)

f_df_target = df[['Adj Close']].to_numpy()

f_df_input = np.array(range(0,len(f_df_target))) # 根据目标数量添加序号
f_df_input = f_df_input.reshape(len(f_df_target),1)

f_train_input, f_test_input, f_train_target, f_test_target = train_test_split(f_df_input, f_df_target , random_state = 42) 

knr = KNeighborsRegressor()
knr.fit(f_train_input, f_train_target)

knr.n_neighbors = 30
print('根据30日均线预测的{}股票在{}的最终成交金额为{}'.format(stock_name,today, knr.predict([[len(f_df_target)+1]])))

plt.plot(f_df_input , f_df_target,color='blue')
plt.plot(f_df_input , knr.predict(f_df_input),color='red')

plt.scatter(len(f_df_target) , knr.predict([[len(f_df_target)+1]]), marker='^')
plt.xlabel('Date')
plt.ylabel('Adj_close')
plt.legend(['k_line','30_line','adj_close'])
plt.show()


##### 用k邻居算法预测各个数据(60日均线)

f_df_target = df[['Adj Close']].to_numpy()

f_df_input = np.array(range(0,len(f_df_target))) # 根据目标数量添加序号
f_df_input = f_df_input.reshape(len(f_df_target),1)

f_train_input, f_test_input, f_train_target, f_test_target = train_test_split(f_df_input, f_df_target , random_state = 42) 

knr = KNeighborsRegressor()
knr.fit(f_train_input, f_train_target)

knr.n_neighbors = 60
print('根据60日均线预测的{}股票在{}的最终成交金额为{}'.format(stock_name,today, knr.predict([[len(f_df_target)+1]])))


plt.plot(f_df_input , f_df_target,color='blue')
plt.plot(f_df_input , knr.predict(f_df_input),color='red')

plt.scatter(len(f_df_target) , knr.predict([[len(f_df_target)+1]]), marker='^')
plt.xlabel('Date')
plt.ylabel('Adj_close')
plt.legend(['k_line','60_line','adj_close'])
plt.show()