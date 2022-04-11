# 1、预载入模块
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras, nn
from tensorflow.keras import layers
from tensorflow.keras.layers import StringLookup, Dense, Concatenate, Input
from keras.utils.vis_utils import plot_model
# from isotonic import Isotonic

# 2、数据处理
raw_data = pd.read_csv("wide_and_deep.csv", header=None)  # ————需加header=None
# print("########################################################################")
# print(raw_data)

# 2.1 one-hot变为字符
# （1）类型1————结果为string类型
soil_type_values = [f"soil_type_{idx + 1}" for idx in range(40)]
# （2）类型2————结果为string类型
wilderness_area_values = [f"area_type_{idx + 1}" for idx in range(4)]
# print(raw_data.loc[:, 14:53])
# (3)类型1————得到一个向量
soil_type = raw_data.loc[:, 14:53].apply(lambda x: soil_type_values[0::1][x.to_numpy().nonzero()[0][0]], axis=1)
# print("########################################################################")
# print("类型变量1： ")
# print(soil_type)
# （4）类型2————得到一个向量
wilderness_area = raw_data.loc[:, 10:13].apply(lambda x: wilderness_area_values[0::1][x.to_numpy().nonzero()[0][0]],axis=1)
# print("类型变量2： ")
# print(wilderness_area)
# type1 = np.asarray(soil_type_values )
#print(type1.transpose())
# 2.2 CSV_HEADER
CSV_HEADER = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Wilderness_Area",
    "Soil_Type",
    "Cover_Type",  # 标签
    "incentive1",  # 激励水平1
    "incentive2",  # 激励水平2
    "incentive3",  # 激励水平3
    "incentive4",
    "incentive5",
    "incentive6",
]
#2.3 最终数据data
# （1）数据合成
data = pd.concat([raw_data.loc[:, 0:9], wilderness_area, soil_type, raw_data.loc[:, 54]], axis=1, ignore_index=True, )
# （2）加入列名
data.columns = CSV_HEADER[0:13]  # 取出前13列
# print("########################################################################")
#print("处理后，将类型变量变为字符")
#print(data)

'''# 3、训练集和测试集的定义
train_splits = []
test_splits = []
# 3.1 初步拆分训练集与测试集
for _, group_data in data.groupby("Cover_Type"):
    random_selection = np.random.rand(len(group_data.index)) <= 0.85  # 生成逻辑变量
    train_splits.append(group_data[random_selection])  # 分散表示
    test_splits.append(group_data[~random_selection])
# 打印分散表示
print("########################################################################")
print("训练集为： ")
print(train_splits)
print("测试集为： ")
print(test_splits)
# 3.2 进一步处理采样并重设
# 处理后的数据为多个[n*13]的三维数据，进行拼接成二维的[m*13]数据
train_data = pd.concat(train_splits).sample(frac=1).reset_index(drop=True)
test_data = pd.concat(test_splits).sample(frac=1).reset_index(drop=True)
print("########################################################################")
print("进一步处理之后： ")
print("训练集为： ")
print(train_data)
print("测试集为： ")
print(test_data)'''
# 3.3 存储
train_data_file = "train_data.csv"
test_data_file = "test_data.csv"
# train_data.to_csv(train_data_file, index=False)
# test_data.to_csv(test_data_file, index=False)

# 4、元数据定义
# 4.1 标签
# （1）名称
# Target_Feature_Name
TARGET_FEATURE_NAME = "Cover_Type"
# （2）变量
# Target_Feature_Labels
TARGET_FEATURE_LABELS = ["0", "1", "2", "3", "4", "5", "6"]  # Target_Feature_Labels
# 4.2 数值特征变量名
# Numeric_Feature_Names
NUMERIC_FEATURE_NAMES = [
    "Aspect",
    "Elevation",
    "Hillshade_3pm",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Horizontal_Distance_To_Fire_Points",
    "Horizontal_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Slope",
    "Vertical_Distance_To_Hydrology",
]
# 4.3 激励水平变量名
INCENTIVE_NAMES = [
    "incentive1",
    "incentive2",
    "incentive3",
    "incentive4",
    "incentive5",
    "incentive6",
]
# 4.4 类别字典，key是："Soil_Type"、"Wilderness_Area"，内容是具体取值
# Categorical_Features_With_Vocabulary
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    # unique:剔除列表中相同的元素返回原排列顺序，在最初excel中不重复的soil one-hot变量第一次出现顺序就是'soil_type_29', 'soil_type_12'···
    "Soil_Type": list(data["Soil_Type"].unique()),
    "Wilderness_Area": list(data["Wilderness_Area"].unique()),
}
# 第一个key对应长度为40的list，第二个key对应长度为4的list
# print("########################################################################")
# print("This is CATEGORICAL_FEATURES_WITH_VOCABULARY")
# # Categorical_Features_With_Vocabulary
# print(CATEGORICAL_FEATURES_WITH_VOCABULARY)
# 4.5 类别特征名————包含两个
#Categorical_Features_Names = list(Categorical_Features_With_Vocabulary.keys())
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())  # Categorical_Feature_Names
# 4.6 所有特征名
# Feature_Names = Numeric_Feature_Names + Categorical_Feature_Names
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
# print(FEATURE_NAMES)  # 12个值
# 4.7 默认值：数值特征、标签默认值设为0，类别默认为'NA'
# Column_Defaults
# Numeric_Feature_Names + [Target_Feature_Name]
COLUMN_DEFAULTS = [
    [0] if feature_name in NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME] + INCENTIVE_NAMES else ["NA"]
    for feature_name in CSV_HEADER
]
# 4.8 N_T_NAME？？？？
print("NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME] + INCENTIVE_NAMES为  ")
N_T_NAME = NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME] + INCENTIVE_NAMES
print(N_T_NAME)
# （10）类别数量
NUM_CLASSES = len(TARGET_FEATURE_LABELS)  # 7

# 5、试验设置
# 5.1 取出以批数据集的函数
#                        文件路径         批的规模     是否shuffle
def get_dataset_from_csv(csv_file_path, batch_size, shuffle=False):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,  # A、路径
        batch_size=batch_size,  # B、批的规模
        column_names=CSV_HEADER,  # C、头（包含标签）
        column_defaults=COLUMN_DEFAULTS,  # D、列默认值
        label_name=TARGET_FEATURE_NAME,  # E、指明标签
        num_epochs=1,  # F、epochs
        header=True,  #
        shuffle=shuffle,  #
    )
    return dataset.cache()  # 返回
# 5.2 参数设置
learning_rate = 0.001  # A、学习率
dropout_rate = 0.1     # B、
batch_size = 265       # C、
num_epochs = 2         # D、epochs
hidden_units = [32]    # , 32]  # 隐藏单元
hidden_units1 = 32


# 6、训练函数
def run_experiment(model):
    # 6.1 模型编译
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),  # A、优化器——设置学习率
        loss=keras.losses.SparseCategoricalCrossentropy(),             # B、损失函数
        metrics=[keras.metrics.SparseCategoricalAccuracy()],           # C、度量
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


# 7、构建模型输入
# 7.1 其它特征的字典
def create_model_inputs():  # 单纯创建len(numeric_feature_names)个inputs输入
    inputs = {}
    # 7.1 特征名循环————共12个：10个数值特征，2个类型特征
    for feature_name in FEATURE_NAMES:
        # （1）数值特征:float类型
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(             # 利用layers.input定义、需要特征名、需要类型、需要shape
                name=feature_name, shape=(), dtype=tf.float32
            )
        # （2）字符特征：string类型
        else:
            inputs[feature_name] = layers.Input(  # 是字符特征时的输入
                name=feature_name, shape=(), dtype=tf.string
            )
    return inputs
# 7.2 等渗嵌入向量的字典
def create_incentive_inputs():
    inputs_iso = {}
    for incentive_name in INCENTIVE_NAMES:
        inputs_iso[incentive_name] = layers.Input(
            name=incentive_name, shape=(), dtype=tf.float32
        )
    return inputs_iso  # n个0或1值

# 8、特征嵌入————StringLookup
# 8.1 其它特征的嵌入
def encode_inputs(inputs, use_embedding=False):
    # 8.1 一开始encoded_features 为list类型
    encoded_features = []
    # 8.2 输入循环————所取出的第一个为“Aspect”
    for feature_name in inputs:
        # 属于类型特征的处理
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            #            Categorical_Features_With_Vocabulary————取出Soil_Type或Wildrness_Area
            # A、根据"Soil_Type"或“Wilderness_Area”创建字典
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            # B、嵌入之one-hot嵌入，若为use_embedding，则返回int类型，若不是use_embedding，则返回binary类型
            # 这里其实是对lookup这函数进行定义
            lookup = StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int" if use_embedding else "binary",  # 嵌入模式则输出模型为int，否则为二进制（one-hot）
            )
            # C、如果是嵌入模型
            if use_embedding:
                # Convert the string input values into integer indices.
                # vocabulary是第几个每种soil数据在最原始数据中第一次出现在40个数据中的位数，'soil_type_29', 'soil_type_12', ···（共40个）
                # lookup是这个数据对应vocabulary中第n个数，就返回n；输入soil_type_n，返回对应的数字顺序，比如n为12，返回2：↑
                # a、先进行one-hot嵌入，在进行训练、预测时，inputs[feature_name]传入具体数值，如soil_type_29
                #    然后lookup根据字典，以及soil_type_29，来生成int类型的one-hot向量
                encoded_feature = lookup(inputs[feature_name])
                print(encoded_feature)
                # b、嵌入维度
                embedding_dims = int(math.sqrt(len(vocabulary)))

                # c、嵌入操作的定义，输入维度为字典维度，输出维度为嵌入维度！
                embedding = layers.Embedding(
                    input_dim=len(vocabulary), output_dim=embedding_dims
                )
                # Convert the index values to embedding representations.
                # d、嵌入操作本身：对one-hot的encoded_feature进行嵌入，并存储在encoded_feature之中
                encoded_feature = embedding( encoded_feature )
                print( encoded_feature )
            else:
            # D、如果无需嵌入——————为什么不需要嵌入的就需要expand_dims？
                # Convert the string input values into a one hot encoding.
                encoded_feature = lookup(  tf.expand_dims(inputs[feature_name], -1)  )
        else:
        # 属于数值特征的处理
            # Use the numerical features as-is.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
        encoded_features.append(encoded_feature)

    all_features = layers.concatenate(encoded_features)
    return all_features
# 8.2 等渗特征的嵌入
def encode_incentive_inputs(inputs_iso):
    encoded_features = []
    for feature_name in inputs_iso:
        encoded_feature = tf.expand_dims(inputs_iso[feature_name], -1)
        encoded_features.append(encoded_feature)

    all_features = layers.concatenate(encoded_features)
    return all_features


# 9、基准模型定义
def create_baseline_model():
    # 1、定义“其它特征”的输入————字典形式
    inputs = create_model_inputs()
    print("##########################################inputs")
    print(inputs)

    # 2、定义“等渗嵌入向量”的输入————字典形式
    inputs_incentive = create_incentive_inputs()

    # 3、“等渗嵌入向量”编码
    feature_incentive = encode_incentive_inputs(inputs_incentive)
    # print(feature_incentive)
    # feature_incentive = Input(shape=(11,), name='incentive')

    # hidden_units = [32, 32], 即units从第一个32for到第二个32，共轮回两次，
    # 4、偏置学习网络 ###########################################################
    #  4.1 其它特征编码
    encode_feature = encode_inputs(inputs)
    # features_bias = encode_inputs(inputs)
    # 4.2 bias网络
    Bias_hidden1 = layers.Dense(hidden_units1 , name = 'Bias_Mlp1')(encode_feature)
    Bias_hidden2 = layers.Dense(hidden_units1, name='Bias_Mlp2')(Bias_hidden1)
    Bias_last = layers.Dense(hidden_units1 , name = 'Bias_Last')(Bias_hidden2)
    features_bias = Dense(1, activation='sigmoid', name='bias_representation')(Bias_last)
    bias_prediction = Dense(1, activation='sigmoid', name='bias_prediction')(Bias_last)

    encode_feature_R = layers.Reshape( (1,54) )(encode_feature)
    encode_feature1 = Concatenate(axis = 1)([encode_feature_R,encode_feature_R])
    #for i in range(5):
    #    encode_feature1.append(encode_feature)

    # 基于Simple的
    out = layers.GRU(32, return_sequences=True, name = 'Simple_RNN')(encode_feature1 )

    out = layers.GRU(32, return_sequences=True, name='Simple_RNN1')(encode_feature1, initial_state=Bias_last)
    weight_out = Dense(1, activation='sigmoid', name='bias_representation')(out)
    model_RNN = keras.models.Model(inputs=inputs, outputs=weight_out)
    # model_RNN.summary()
    plot_model(model_RNN, to_file='model_RNN111.png', show_shapes=True, rankdir="LR")

    a = 1

    # for units in hidden_units:
    #     features_bias = layers.Dense(units, name='biasMLP1')(encode_feature)
    #     features_bias = layers.BatchNormalization(name='biasMLP2')(features_bias)
    #     features_bias = layers.ReLU(name='biasMLP3')(features_bias)
    #     # dropout:防止过拟合，所有元素按照1/(1-rate)的比例扩大，此处为0.1
    #     features_bias = layers.Dropout(dropout_rate, name='biasMLP4')(features_bias)
    # features_bias = Dense(1, activation='LeakyReLU', name='bias_representation')(features_bias)
    # bias_prediction = Dense(1, activation='sigmoid', name='bias_prediction')(features_bias)




    # unlift #######################################################
    # features_unlift = encode_inputs(inputs)
    for units in hidden_units:
        features_unlift = layers.Dense(units, name='upliftMLP1')(encode_feature)
        features_unlift = layers.BatchNormalization(name='upliftMLP2')(features_unlift)
        features_unlift = layers.ReLU(name='upliftMLP3')(features_unlift)
        # dropout:防止过拟合，所有元素按照1/(1-rate)的比例扩大，此处为0.1
        features_unlift = layers.Dropout(dropout_rate, name='upliftMLP4')(features_unlift)
    features_unlift = Dense(5, activation='LeakyReLU', name='weight_representation')(features_unlift)

    # concatenate bias and uplift weight ################################
    # concatenate = Concatenate(axis=-1)([features_bias, features_unlift])
    concatenate = Concatenate(axis=-1)([features_bias, features_unlift])
    # print(concatenate)

    # inner product part
    inner = layers.Dot(axes=1, name='inner_product')([concatenate, feature_incentive])

    prediction = keras.activations.sigmoid(inner)

    # num_classes = 7
    #model = keras.models.Model(inputs=[inputs, inputs_incentive], outputs=[prediction, bias_prediction])
    model = keras.models.Model(inputs=[inputs, inputs_incentive], outputs=prediction)
    model.summary()
    plot_model(model, to_file='DNN_model1.png',show_shapes=True, rankdir="LR")

    return model


# 10、创建基准模型
baseline_model = create_baseline_model()
# plot_model(baseline_model, to_file='DIPN_multiinput.png', show_shapes=True, rankdir="LR")

# 11、运行基准模型
#run_experiment(baseline_model)
