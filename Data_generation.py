###############################################################################################
####    一、预载入模块    ########################################################################
###############################################################################################
import random
import math
import numpy as np
import pandas as pd


# 特征的取值
n1_feature = [1,2,3]
n2_feature = [1,2,3,4]
n3_feature = [1,2,3,4]
n4_feature = [1,2]

# 特征名称
columns1 = ["n1","n2","n3","n4","promotion","Delta_CTR"] # Delta_CTR = CTR增量

# 数据生成
Num_of_Sample_Matirx = []
DataSet = pd.DataFrame()

for n1 in n1_feature:
    for n2 in n2_feature:
        for n3 in n3_feature:
            for n4 in n4_feature:
                # （1）固定参数设定########################################
                a = random.uniform(0, 1)  # a、
                b = 1 - a  # b、
                Miu = random.uniform(-50, 150)  # c、
                Delta = random.uniform(0, 50)  # d、
                Z_All = []  # 每次的Miu是不同的
                # （2）DIPN论文公式（20）中Z的计算 #########################
                for h in range(101):
                    Base1 = -(h - Miu) ** 2 / 2 / Delta ** 2
                    Z_All.append(math.exp(Base1))
                Z = max(Z_All)
                if ( Z == 0 ):
                    Z = 0.0001

                # （3）子样本总数###########################################
                Num_of_Sample = random.randint(1, 1000)     # a、总量
                Num_of_Sample_Matirx.append(Num_of_Sample)  # b、记录子样本总数
                Sub_DataSet = pd.DataFrame()                # c、子样本DataFrame定义

                # （4）子样本生成###########################################
                i = -1
                for sample_index in range(Num_of_Sample):
                    p = random.randint(1, 100)  # 1~100  # a、提升系数的生成
                    sum_z = 0.0                          # b、和
                    for h2 in range(p+1):                # c、循环求和
                        Base2 = -(h2 - Miu) ** 2 / 2 / Delta ** 2   # （a）指数函数的指数
                        sum_z = sum_z + math.exp(Base2)             # （b）求和
                    Response_p_i = a + b / 100 / Z * sum_z          # （c）响应
                                                          # d、以Response_p_i为概率选择是否响应
                    click = np.random.choice([0, 1], size=1, p=[1 - Response_p_i, Response_p_i])
                    meta_data = {
                        "n1": [n1],
                        "n2": [n2],
                        "n3": [n3],
                        "n4": [n4],
                        "promotion": [p],
                        "Delta_CTR": [Response_p_i],
                        "label": click
                    }
                    i = i + 1
                    j = [i]
                    # 数据载入到 DataFrame 对象
                    sample_itme = pd.DataFrame(meta_data, index=j)
                    # 加入到子数据集
                    Sub_DataSet = Sub_DataSet.append(sample_itme)
                DataSet = DataSet.append(Sub_DataSet).reset_index(drop=True)
DataSet.to_csv("Excite.csv", index=False)
a = 1