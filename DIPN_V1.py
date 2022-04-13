###############################################################################################
# 一、预载入模块
###############################################################################################
# 1.1 python
import math
import numpy as np
import pandas as pd
# 1.2 tensor_flow的模块
import tensorflow as tf
from tensorflow import keras, nn
# 1.3 keras的模块
from tensorflow.keras import layers
from tensorflow.keras.layers import StringLookup, IntegerLookup, Dense, Concatenate, Input
# 1.4 作图
from keras.utils.vis_utils import plot_model


###############################################################################################
# 二、数据载入与训练集和测试集划分
###############################################################################################
# 1、读取数据
raw_data = pd.read_csv("Excite1.csv")  # ————需加header=None
# print("########################################################################")
# print(raw_data)

# 2、读取数据的头
Csv_Haeder = list(raw_data.columns)
print("数据集的头为")
print( Csv_Haeder )

# 3、训练集和测试集的定义
train_splits = []
test_splits = []
# 3.1 初步拆分训练集与测试集
for _, group_data in raw_data.groupby("label"):
    random_selection = np.random.rand(len(group_data.index)) <= 0.85  # 生成逻辑变量
    train_splits.append(group_data[random_selection])  # 分散表示
    test_splits.append(group_data[~random_selection])
# 3.2 进一步处理采样并重设
train_data = pd.concat(train_splits).sample(frac=1).reset_index(drop=True)
test_data = pd.concat(test_splits).sample(frac=1).reset_index(drop=True)

# 3.3 存储
train_data_file = "train_data_excite.csv"
test_data_file = "test_data_excite.csv"
train_data.to_csv(train_data_file, index=False)
test_data.to_csv(test_data_file, index=False)

###############################################################################################
# 三、元数据定义
###############################################################################################
# 1、标签
# 1.1 名称
Target_Feature_Name = "label"
# 1.2 取值
Target_Feature_Labels = [0, 1]

# 2、特征名
Feature_Names_Other = ["category_1", "category_2", "category_3", "category_4"]
Feature_Names_Promotion = ["promotion" ]
# 3、类别变量字典
Categorical_Features_With_Vocabulary = {
    "category_1": list(raw_data["category_1"].unique()),
    "category_2": list(raw_data["category_2"].unique()),
    "category_3": list(raw_data["category_3"].unique()),
    "category_4": list(raw_data["category_4"].unique()),
}

Num_Classes= len( Target_Feature_Labels )  # 7
###############################################################################################
# 三、元数据定义
###############################################################################################
# 1 取出以批数据集的函数
#                        文件路径         批的规模     是否shuffle
def get_dataset_from_csv(csv_file_path, batch_size, shuffle=False):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,          # A、路径
        batch_size=batch_size,  # B、批的规模
        column_names= Csv_Haeder,  # C、头（包含标签）
        # column_defaults=COLUMN_DEFAULTS,  # D、列默认值
        label_name = Target_Feature_Name,  # E、指明标签
        num_epochs=1,  # F、epochs
        header=True,  #
        shuffle=shuffle,  #
    )
    return dataset.cache()  # 返回
# 2 参数设置
learning_rate = 0.001  # A、学习率
dropout_rate = 0.1     # B、
batch_size = 265       # C、
num_epochs = 2         # D、epochs
# hidden_units = [32]    # , 32]  # 隐藏单元
hidden_units1 = 32
# 3、训练函数
def run_experiment(model):
    # 6.1 模型编译
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),  # A、优化器——设置学习率
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    # 6.2 训练集
    train_dataset = get_dataset_from_csv(train_data_file, batch_size, shuffle=True)
    print( train_dataset )
    # 6.3 测试集
    test_dataset = get_dataset_from_csv(test_data_file, batch_size)
    # 6.4 训练
    print("Start training the model...")
    history = model.fit(train_dataset, epochs=num_epochs)  # 自带循环
    print("Model training finished")
    # 6.5 返回结果————测试
    _, accuracy = model.evaluate(test_dataset, verbose=0)

    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

###############################################################################################
# 四、嵌入层
###############################################################################################
# 1、构建其它特征的输入：shape构建为()的形式
def create_model_inputs_other():
    inputs = {}
    for feature_name in Feature_Names_Other:
        inputs[feature_name] = layers.Input(name=feature_name, shape=(), dtype=tf.float32 )
    return inputs
# 2、构建激励特征的输入
def create_model_inputs_Promotion():
    input = {}
    input["promotion"] = layers.Input(name="promotion", shape=(), dtype=tf.float32 )
    return input

# 3、类别特征的one-hot
def encode_Category(inputs):
    encoded_features = []
    for feature_name in inputs:
        vocabulary = Categorical_Features_With_Vocabulary[ feature_name ]
        lookup = IntegerLookup(
            vocabulary=vocabulary,
            mask_token=None,
            num_oov_indices=0,
            output_mode="int",
        )
        encoded_feature = lookup(inputs[feature_name])    # one-hot
        embedding_dims = int(math.sqrt(len(vocabulary)))  # 进一步嵌入的维度
        embedding = layers.Embedding(input_dim=len(vocabulary), output_dim=embedding_dims)  # 进一步嵌入的函数
        encoded_feature = embedding(encoded_feature)                                        # 进一步嵌入,当前输出结果为(None,2,1)
        encoded_features.append(encoded_feature)    # 添加！
    all_features = layers.concatenate(encoded_features)
    return all_features
# 4、等渗嵌入
def encode_Promotion(inputs):
    encoded_features = []
    inputs_iso = inputs["promotion"]
    for i in range( 100 ):
        level = tf.constant(i, dtype=tf.float32)                  # A、比较标准
        encoder_bit = tf.math.greater(inputs_iso, level)          # B、逐个比较
        encoder_bit = tf.cast(encoder_bit, tf.float32, name=None) # C、类型转换
        encoded_bit_expDims = tf.expand_dims(encoder_bit, -1)     # D、扩展维度
        encoded_features.append(encoded_bit_expDims)              # E、加入该位
        a = 1
    all_features = layers.concatenate(encoded_features)           # F、合并返回
    a = 1
    return all_features

###############################################################################################
# 五、模型定义
###############################################################################################
def create_model():
    # 1、输入定义
    inputs_Category = create_model_inputs_other()
    inputs_Promotion = create_model_inputs_Promotion()

    # 2、特征嵌入
    encode1_Category  = encode_Category(inputs_Category)
    encode1_Promotion = encode_Promotion(inputs_Promotion)
    # 3、bias_net ###########################################################
    Bias_hidden = layers.Dense(hidden_units1 , name = 'Bias_Mlp1')(encode1_Category)
    Bias_hidden = layers.Dense(hidden_units1, name='Bias_Mlp2')(Bias_hidden)
    Bias_hidden = layers.Dense(hidden_units1 , name = 'Bias_Last')(Bias_hidden)
    # features_bias = Dense(1, activation='sigmoid', name='bias_representation')(Bias_hidden)
    bias_prediction = Dense(1, activation='sigmoid', name='bias_prediction')(Bias_hidden) # 维度为(None,1)

    # 4、uplift-net
    encode_feature_R = layers.RepeatVector(100)(encode1_Category)   # 4.1 输入——RepeatVector
    out = layers.GRU(32, return_sequences=True, name='Simple_RNN1')(encode_feature_R, initial_state=Bias_hidden)
    weight_out = Dense(1, activation='sigmoid', name='bias_representation')(out)  # 维度为（None,100,1）
    weight_out1 = layers.Reshape((100,))(weight_out)

    # 5、乘积以预测响应
    W_dot_E = layers.Dot(axes=1, name='inner_product')([weight_out1, encode1_Promotion])
    W_dot_E_add_B = layers.Add()([W_dot_E,bias_prediction])
    prediction = keras.activations.sigmoid(W_dot_E_add_B)

    model_RNN = keras.models.Model(inputs=encode1_Category, outputs=weight_out)   #
    # model_RNN.summary()
    plot_model(model_RNN, to_file='model_RNN111.png', show_shapes=True, rankdir="LR")
    model = keras.models.Model(inputs=[inputs_Category, inputs_Promotion], outputs=prediction)
    #plot_model(model, to_file='Dipn_Final.png',show_shapes=True, rankdir="LR")
    return model
###############################################################################################
# 六、创建模型
###############################################################################################
baseline_model = create_model()

###############################################################################################
# 七、训练模型
###############################################################################################
run_experiment(baseline_model)
