# 🐱🐶 猫狗分类深度学习项目

一个使用自定义卷积神经网络（CNN）进行猫狗图像分类的深度学习项目，无需预训练模型即可达到较高准确率。项目包含完整的数据集、训练代码、预训练模型和测试工具。

## 📋 项目概述

本项目实现了一个完整的深度学习流水线，用于对猫和狗的图像进行分类。项目从零开始构建了一个多层卷积神经网络，通过数据增强、正则化和优化技术提高模型性能。项目已包含预训练模型，用户可直接用于推理和评估。

### ✨ 主要特点
- 🎯 **从头训练**：不使用预训练网络，完全自定义CNN架构
- 🔧 **修复版本**：解决了`analyzeNetwork`冲突问题
- 📊 **全面评估**：提供多种评估指标和可视化工具
- 🛠️ **模块化设计**：代码结构清晰，易于修改和扩展
- 📈 **性能优化**：通过数据增强、批归一化和dropout提高泛化能力
- 🤖 **预训练模型**：包含训练好的模型，可直接使用
- 📝 **实验报告**：详细的实验结果和分析报告
- 🧪 **测试工具**：提供专用测试脚本进行模型验证

## 📁 项目结构

```
cat_dog_classification/
├── cat_dog_classification.m      # 主训练程序文件
├── test_plus.m                   # 增强测试脚本
├── cat_dog_classifier_fixed.mat  # 预训练模型（已包含）
├── experiment_summary_fixed.txt  # 实验总结报告（已包含）
├── experiment_results_fixed.mat  # 实验结果数据（训练后生成）
├── README.md                     # 项目说明文档
├── 数据集/                       # 猫狗分类数据集
│   ├── train/                    # 训练集
│   │   ├── cats/                 # 猫类图像
│   │   └── dogs/                 # 狗类图像
│   ├── val/                      # 验证集
│   └── test/                     # 测试集
└── 文档/                         # 项目相关文档
    ├── 模型架构图.png            # 网络结构可视化
    └── 实验结果/                 # 训练过程中生成的可视化结果
```

## 🔧 环境要求

### 必需软件
- MATLAB R2020a或更高版本
- Deep Learning Toolbox
- Computer Vision Toolbox

### 推荐硬件
- 支持CUDA的NVIDIA GPU（推荐，用于加速训练）
- 至少8GB RAM
- 10GB可用磁盘空间

## 📊 数据集

### 数据集来源
本项目使用的数据集来自 **ModelScope社区**，具体为：
- **数据集名称**: cat_vs_dog_class
- **来源链接**: [https://www.modelscope.cn/datasets/XCsunny/cat_vs_dog_class/summary](https://www.modelscope.cn/datasets/XCsunny/cat_vs_dog_class/summary)
- **数据集描述**: 包含大量猫和狗的分类图像，适用于二分类任务

### 数据集统计
- **总图像数**: 约10,000张
- **猫类图像**: 约5,000张
- **狗类图像**: 约5,000张
- **图像尺寸**: 大小不一，程序会自动调整为224×224
- **格式**: JPEG、PNG等常见图像格式

### 数据划分
- 训练集：60%（约6000张）
- 验证集：20%（约2000张）
- 测试集：20%（约20500张）

## 🤖 预训练模型

### 模型文件
- **文件名**: `cat_dog_classifier_fixed.mat`
- **文件大小**: 约150MB
- **包含内容**: 
  - 训练好的神经网络模型 (`net`)
  - 训练过程信息 (`trainInfo`)
  - 模型架构和参数

### 模型性能
基于测试集的评估结果：
- **准确率**: 91.5%
- **精确率**: 92.3%
- **召回率**: 90.8%
- **F1分数**: 91.5%
- **AUC**: 0.956

### 模型使用
```matlab
% 加载预训练模型
load('cat_dog_classifier_fixed.mat', 'net');

% 对新图像进行分类
img = imread('test_image.jpg');
imgResized = imresize(img, [224 224]);
[label, score] = classify(net, imgResized);

fprintf('预测结果: %s (置信度: %.2f%%)\n', char(label), max(score)*100);
```

## 🚀 快速开始

### 方法一：直接使用预训练模型
1. 下载本项目所有文件
2. 运行增强测试脚本：
   ```matlab
   run test_plus.m
   ```
3. 脚本将自动加载预训练模型并在测试集上评估性能

### 方法二：重新训练模型
1. 确保数据集路径正确（默认：`D:\深度学习`）
2. 运行主训练程序：
   ```matlab
   run cat_dog_classification.m
   ```
3. 程序将训练新模型并保存结果

### 方法三：使用自己的图像测试
1. 将待分类图像放入任意目录
2. 使用以下代码进行分类：
   ```matlab
   % 加载模型
   load('cat_dog_classifier_fixed.mat', 'net');
   
   % 处理单张图像
   imgPath = 'your_image.jpg';
   img = imread(imgPath);
   imgResized = imresize(img, [224 224]);
   
   % 进行分类
   [label, scores] = classify(net, imgResized);
   fprintf('分类结果: %s\n', char(label));
   ```

## 📝 实验报告

### 报告文件
- **文件名**: `experiment_summary_fixed.txt`
- **生成时间**: 训练完成后自动生成
- **内容概述**:
  - 数据集统计信息
  - 训练参数配置
  - 网络架构描述
  - 详细性能指标
  - 混淆矩阵结果

### 报告摘要
```
猫狗分类实验报告（修复版）
===============================

数据集信息:
  训练集: 6000 张图片
  验证集: 2000 张图片
  测试集: 2000 张图片

性能指标:
  测试集准确率: 91.50%
  验证集准确率: 92.20%
  精确率 (Precision): 92.30%
  召回率 (Recall): 90.80%
  F1分数: 91.50%
  AUC: 0.956
```

## 🧪 测试脚本：test_plus.m

### 功能特点
`test_plus.m` 是一个增强测试脚本，提供以下功能：

1. **模型加载与验证**: 自动加载预训练模型并验证完整性
2. **批量测试**: 对整个测试集进行批量分类
3. **性能分析**: 计算多种评估指标
4. **可视化**: 生成混淆矩阵、ROC曲线等可视化结果
5. **错误分析**: 识别并展示分类错误的样本
6. **实时推理**: 支持对新图像的实时分类

### 使用方法
```matlab
% 运行增强测试脚本
test_plus;

% 或者带参数调用
test_plus('show_misclassified', true, 'num_samples', 20);
```

### 参数配置
```matlab
% 测试参数设置
params.testFolder = 'D:\深度学习\test';  % 测试集路径
params.batchSize = 32;                   % 批量大小
params.showResults = true;               % 显示可视化结果
params.saveResults = true;               % 保存测试结果
```

## 🏗️ 模型架构

### 网络结构
```
输入层 (224×224×3)
│
├─ 卷积块1 (3×3, 32 filters, BN, ReLU, MaxPool)
├─ 卷积块2 (3×3, 64 filters, BN, ReLU, MaxPool)
├─ 卷积块3 (3×3, 128 filters, BN, ReLU, MaxPool)
├─ 卷积块4 (3×3, 256 filters, BN, ReLU, MaxPool)
│
├─ 全连接层1 (512 units, BN, ReLU, Dropout 0.5)
├─ 全连接层2 (256 units, BN, ReLU, Dropout 0.3)
│
└─ 输出层 (2 units, Softmax, Classification)
```

### 关键技术
- **批归一化**: 加速训练并提高稳定性
- **Dropout**: 防止过拟合 (0.5和0.3的dropout率)
- **数据增强**: 随机旋转(±30°)、翻转、缩放(0.7-1.3)、平移(±30像素)
- **学习率调度**: 分段学习率衰减（每10轮减半）

## ⚙️ 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 图像尺寸 | 224×224×3 | 输入图像大小 |
| 批量大小 | 32 | 每次迭代处理的样本数 |
| 训练轮数 | 50 | 最大训练轮数 |
| 初始学习率 | 0.001 | Adam优化器初始学习率 |
| 学习率衰减 | 每10轮减半 | 分段学习率调度 |
| L2正则化 | 0.0005 | 权重衰减系数 |
| 验证频率 | 每30次迭代 | 验证集评估频率 |

## 📈 性能评估

### 评估指标
- ✅ 准确率 (Accuracy): 91.5%
- ✅ 精确率 (Precision): 92.3%
- ✅ 召回率 (Recall): 90.8%
- ✅ F1分数 (F1-Score): 91.5%
- ✅ AUC值 (Area Under Curve): 0.956
- ✅ 混淆矩阵 (Confusion Matrix): 详细分类结果

### 可视化输出
程序生成以下可视化图表：
1. **训练过程**: 训练/验证准确率和损失曲线
2. **混淆矩阵**: 测试集和验证集的分类结果
3. **预测示例**: 随机样本的预测结果可视化
4. **ROC曲线**: 接收者操作特征曲线
5. **性能指标**: 详细数值指标

## 🔍 故障排除

### 常见问题

1. **模型加载失败**
   ```matlab
   % 错误信息: Unable to read file
   % 解决方案: 确保文件路径正确，或重新下载模型文件
   ```

2. **内存不足错误**
   - 减小批量大小（从32减少到16）
   - 减小图像尺寸（从224×224减小到128×128）

3. **分类结果不准确**
   - 确保输入图像为RGB格式
   - 图像大小调整为224×224
   - 检查图像质量（清晰度、对比度）

### MATLAB版本兼容性
- 最低要求: MATLAB R2020a
- 推荐版本: MATLAB R2021b或更高
- 工具箱要求: Deep Learning Toolbox, Computer Vision Toolbox

## 🧪 高级功能

### 集成学习
项目包含可选的集成学习功能，通过训练多个不同架构的模型并采用投票法集成：
```matlab
% 在训练过程中选择集成学习
是否要训练多个模型进行集成？(y/n): y
```
- 训练3个不同架构的模型
- 采用多数投票法进行集成
- 通常能提高1-3%的准确率

### 模型微调
```matlab
% 加载预训练模型进行微调
load('cat_dog_classifier_fixed.mat', 'net');
layers = net.Layers;

% 修改最后一层进行新任务微调
layers(end-2) = fullyConnectedLayer(10); % 10类分类
layers(end) = classificationLayer;

% 继续训练
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.0001, ... % 较小的学习率
    'MaxEpochs', 20);
```

## 📚 参考文献与数据来源

1. **数据集来源**: 
   - ModelScope猫狗分类数据集: [https://www.modelscope.cn/datasets/XCsunny/cat_vs_dog_class/summary](https://www.modelscope.cn/datasets/XCsunny/cat_vs_dog_class/summary)

2. **技术文献**:
   - Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks.
   - Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition.
   - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.

## 📄 许可证

本项目代码遵循MIT许可证。数据集使用请遵循ModelScope平台的相应使用条款。

## 🤝 贡献

欢迎提交问题和改进建议！

### 如何贡献
1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个Pull Request

## 📧 联系方式

如有问题或建议，请通过GitHub Issues提交。

---

**重要提示**: 
- 使用数据集时请遵守ModelScope平台的使用条款
- 项目默认使用`D:\深度学习`作为数据路径，请根据实际情况修改
- 预训练模型可直接用于推理，无需重新训练
- 使用`test_plus.m`脚本可快速验证模型性能

**更新日志**:
- v1.0: 初始版本，包含完整训练代码和数据集
- v1.1: 添加预训练模型和测试脚本
- v1.2: 修复analyzeNetwork问题，改进模型架构
- v1.3: 添加集成学习功能和详细实验报告
