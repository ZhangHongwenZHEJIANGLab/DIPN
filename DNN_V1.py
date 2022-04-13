###############################################################################################
# 一、预载入
###############################################################################################
# 1.1 python
import math
import numpy as np
import pandas as pd
# 1.2 tensor_flow
import tensorflow as tf
from tensorflow import keras, nn
# 1.3 keras
from tensorflow.keras import layers
from tensorflow.keras.layers import StringLookup, IntegerLookup, Dense, Concatenate, Input
# 1.4 作图
from keras.utils.vis_utils import plot_model


###############################################################################################
# 二、数据载入与训练集和测试集划分
###############################################################################################
# 1、读取数据
raw_data = pd.read_csv("Excite1.csv")  # ————需加header=None
# 2、读取数据的头
Csv_Haeder = list(raw_data.columns)
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
Feature_Names = Csv_Haeder
Feature_Names = Feature_Names[:5]
Categorical_Feature_Names = Feature_Names[:4]
# 3、类别变量字典
Categorical_Features_With_Vocabulary = {
    "category_1": list(raw_data["category_1"].unique()),
    "category_2": list(raw_data["category_2"].unique()),
    "category_3": list(raw_data["category_3"].unique()),
    "category_4": list(raw_data["category_4"].unique()),
}
###############################################################################################
# 四、数据打包、模型训练等
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
batch_size = 50       # C、
num_epochs = 100        # D、epochs
# hidden_units = [32]    # , 32]  # 隐藏单元
hidden_units1 = 32
# 3、训练函数
def run_experiment(model):
    # 3.1 模型编译
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),  # A、优化器——设置学习率
        loss=keras.losses.BinaryCrossentropy(),                        # B、损失函数
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],       # C、度量
    )
    # 3.2 训练集
    train_dataset = get_dataset_from_csv(train_data_file, batch_size, shuffle=True)
    # 3.3 测试集
    test_dataset = get_dataset_from_csv(test_data_file, batch_size)
    # 3.4 训练
    print("Start training the model...")
    history = model.fit(train_dataset, epochs=num_epochs)  # 自带循环
    print("Model training finished")
    # 6.5 返回结果————测试
    _, accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

###############################################################################################
# 五、嵌入层
###############################################################################################
# 1、构建其它特征的输入：shape构建为()的形式
def create_model_inputs():
    inputs = {}
    for feature_name in Feature_Names:
        inputs[feature_name] = layers.Input(name=feature_name, shape=(), dtype=tf.float32 )
    return inputs
# 2、类别特征的one-hot————不进行嵌入，直接
def encode_inputs(inputs):
    encoded_features = []
    for feature_name in inputs:
        if feature_name in Categorical_Feature_Names:
            vocabulary = Categorical_Features_With_Vocabulary[ feature_name ]
            lookup = IntegerLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                output_mode="binary",
            )
            encoded_feature = lookup(tf.expand_dims(inputs[feature_name], -1))
        else:
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
        encoded_features.append(encoded_feature)    # 添加！
    all_features = layers.concatenate(encoded_features)
    return all_features

###############################################################################################
# 五、模型定义
###############################################################################################
def create_model():
    # 1、输入定义
    inputs_Category = create_model_inputs()
    # 2、特征嵌入
    encode_Category  = encode_inputs(inputs_Category)
    # 3、多层MLP
    Bias_hidden = layers.Dense(hidden_units1 , name = 'Bias_Mlp1')(encode_Category)
    Bias_hidden = layers.Dense(hidden_units1, name='Bias_Mlp2')(Bias_hidden)
    Bias_hidden = layers.Dense(hidden_units1 , name = 'Bias_Last')(Bias_hidden)
    prediction = Dense(1, activation='sigmoid', name='click_prediction')(Bias_hidden) # 维度为(None,1)

    model = keras.models.Model(inputs=inputs_Category, outputs=prediction)   #
    # model_RNN.summary()
    #plot_model(model, to_file='DNN_V1.png', show_shapes=True, rankdir="LR")
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
