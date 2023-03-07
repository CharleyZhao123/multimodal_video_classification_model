# multimodal_video_classification_model
多模态视频分类基础模型

1. 文件说明
dataset.py 和datasets/Brain4cars.py 定义了dataloader
dataset/annotation中fold.csv为各fold的数据划分
levellstm.py 为主模型py文件
mean.py, spatial_transforms.py, temporal_transforms.py为数据预处理文件，功能分别为均值化，空间变换，时间变换
resnet.py 和core文件夹为道路图像和人脸图像的特征提取模型
test.py 和test.sh 为测试用py文件和脚本
train.py 和train.sh 为训练用py文件和脚本

2. 使用说明
先把数据集放到当前目录下
训练：./train.sh
测试：./test.sh

3. 脚本说明

------train.sh--------
root_path为当前工作目录
video_path为当前数据集路径 (数据集在工作目录下)
annotation_path为数据划分文件路径
result_path为模型结果存储路径
dataset 默认为Brain4cars_Unit 即内外视频和生理数据均使用
checkpoint 为每隔多少epoch存一次模型
n_epochs 为模型训练的epoch数
sample_duration 为采样的序列长度，10即采的序列长为10
end_action和interval没用
n_scales图像放缩几次，默认为1不用管
learning_rate 学习率
n_fold 训第几个fold
train_from_scratch是否接着训练，0为否 （建议不用，可能有bug）

--------test.sh-------
和train.sh差不多
test_start_epoch和test_end_epoch为测试的开始和结束epoch

正常训练测试是5折交叉验证，目前是单折，fold0。
