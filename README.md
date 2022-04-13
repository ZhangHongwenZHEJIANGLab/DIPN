# DIPN
* 1、Data_generation.py为数据生成函数

* 2、DIPN_V1为第一版的DIPN，在该版本中，对类别特征进行one-hot之后，还进行了embedding

* 3、DIPN_V2为第二版的DIPN，在该版本中，对类别特征进行one-hot之后，没进行embedding，
主要是考虑到每一个类别特征的one-hot后的维度很小，再进行embedding可能会损失很多信息

* 4、DNN_V1为只用MLP进行拟合的结果，其中将激励水平作为一个数值特征

* 5、DIPN_V3为第三办的DIPN，将其中的GRU换为MLP，待完成
