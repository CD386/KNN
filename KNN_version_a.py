import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

##这个版本的KNN采用欧式距离排序
##权重取欧式距离的倒数为比例
##m和K取遍历 但是很显然遍历无法取到最好 应当选择更优方式选择m和K

##距离计算方式和权重计算方式可以改进

def distance(x,y):
    t=0
    for i in range(len(x)):
        if i >= 2:
            t+=pow((x[i]-y[i]),2)
    return pow(t,1/2)
def takeSecond(elem):
    return elem[1]
def prediction(lists):
    power=0
    result=0
    for i in lists:
        if i[-1]==0:
            return round(i[0],2)
    for i in lists:
        power+=1/i[-1]
    for i in lists:
        result+=i[0]*(1/i[-1])/power
    return round(result,2)

def KNN(data, m, k, circleLenth):
    ##data:目标数据集 m:相空间维度 k:近邻数量 circleLenth:预测周期（从过去多久的数据集中选择近邻）
    start = time.time ()
    if circleLenth < m+k:
        print('预测周期必须比相空间维度和近邻数量的和还大')
        return 0
    listName=data.columns.tolist()
    listName.append('predicted_value')
    listName.append('rate_of_error(%)')
#     print(listName)

    tempData=data.values
    data=[]
    for i in range(len(tempData)):
        tp=list(tempData[i])
        if i >=m:
            for j in range(m):
                tp.append(tempData[i-j-1][-1])
            data.append(tp)
        else:
            for j in range(m):
                tp.append(None)
            data.append(tp)
    for i in range(len(data)):
        if i >= circleLenth+m:
#             print('✅下面一行可以被预测')
            tpt=[]
            for j in range(circleLenth):
                tp=[]
                tp.append(data[i-j-1][1])
                tp.append(distance(data[i],data[i-j-1]))
                tpt.append(tp)
            tpt.sort(key=takeSecond)
            tpt=tpt[:k]
            data[i].append(prediction(tpt))
#         else:
#             print('❌下面一行无法被预测')
#         print(data[i])
#         print('-------------')
    for i in range(len(data)):
        if i >= circleLenth:
            temp=data[i][0:2]
            temp.append(data[i][-1])
            error=round(100*(data[i][-1]-data[i][1])/data[i][1],4)
            if error<0:
                error=-error
            temp.append(error)
            data[i]=temp
        else:
            temp=data[i][0:2]
            temp.append(None)
            temp.append(None)
            data[i]=temp
    data=pd.DataFrame(data,columns=listName)
    end = time.time ()
    print('计算完成')
    print('用时',str(end-start),'s')
    return(data)    



if __name__=="__main__":
    data = pd.read_csv('train26803.csv')
    data = KNN(data,18,14,288)
    print(data.iloc[285:300])
    ##KNN（原数据集，维数，近邻数，预测周期）
    ##预测数据与原数据已集成在返回的数据表中，可以据此进行进一步加工
