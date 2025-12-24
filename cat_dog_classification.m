%% 不使用预训练网络提高正确率的猫狗分类（修复analyzeNetwork问题）
clear; clc; close all;

%% 1. 设置路径和参数
mainFolder = 'D:\深度学习';
trainFolder = fullfile(mainFolder, 'train');
valFolder = fullfile(mainFolder, 'val');
testFolder = fullfile(mainFolder, 'test');

% 图像尺寸
imageSize = [224, 224, 3];

% 训练参数
miniBatchSize = 32;
maxEpochs = 50;
initialLearnRate = 0.001;
validationFrequency = 30;

%% 2. 创建图像数据存储
imdsTrain = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsValidation = imageDatastore(valFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsTest = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% 显示数据集信息
disp('=== 数据集信息 ===');
disp(['训练集: ' num2str(numel(imdsTrain.Files)) ' 张图片']);
disp(['验证集: ' num2str(numel(imdsValidation.Files)) ' 张图片']);
disp(['测试集: ' num2str(numel(imdsTest.Files)) ' 张图片']);

%% 3. 数据增强
% 创建增强策略
augmenter1 = imageDataAugmenter( ...
    'RandRotation', [-30, 30], ...
    'RandXReflection', true, ...
    'RandScale', [0.7, 1.3], ...
    'RandXTranslation', [-30, 30], ...
    'RandYTranslation', [-30, 30]);

% 创建增强图像数据存储
augimdsTrain = augmentedImageDatastore(imageSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter1, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsValidation = augmentedImageDatastore(imageSize(1:2), imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsTest = augmentedImageDatastore(imageSize(1:2), imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

%% 4. 设计改进的神经网络架构
% 创建一个简单但有效的CNN，避免analyzeNetwork冲突
layers = [
    % 输入层
    imageInputLayer(imageSize, 'Name', 'input', 'Normalization', 'zerocenter')
    
    % 卷积块1
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    % 卷积块2
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
    % 卷积块3
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
    
    % 卷积块4
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool4')
    
    % 全连接层
    fullyConnectedLayer(512, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    dropoutLayer(0.5, 'Name', 'dropout1')
    
    fullyConnectedLayer(256, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn6')
    reluLayer('Name', 'relu6')
    dropoutLayer(0.3, 'Name', 'dropout2')
    
    % 输出层
    fullyConnectedLayer(2, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];

% 修复：创建一个不包含输出层的网络用于分析
layersForAnalysis = layers(1:end-1); % 移除分类层

disp('=== 网络架构（不包含输出层）===');
analyzeNetwork(layersForAnalysis);

disp('=== 完整网络架构信息 ===');
disp(['网络层数: ' num2str(numel(layers))]);
for i = 1:numel(layers)
    fprintf('层 %d: %s\n', i, class(layers(i)));
end

%% 5. 训练选项
options = trainingOptions('adam', ...
    'InitialLearnRate', initialLearnRate, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', validationFrequency, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'VerboseFrequency', 20, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'L2Regularization', 0.0005, ...
    'OutputNetwork', 'best-validation-loss', ...
    'CheckpointPath', tempdir);

%% 6. 训练网络
disp('=== 开始训练 ===');
[net, trainInfo] = trainNetwork(augimdsTrain, layers, options);

% 保存训练好的网络
save('cat_dog_classifier_fixed.mat', 'net', 'trainInfo');

%% 7. 评估模型性能
% 在验证集上评估
disp('=== 验证集评估 ===');
[YPredVal, probsVal] = classify(net, augimdsValidation);
YValidation = imdsValidation.Labels;

accuracyVal = sum(YPredVal == YValidation) / numel(YValidation);
disp(['验证集准确率: ', num2str(accuracyVal * 100), '%']);

% 在测试集上评估
disp('=== 测试集评估 ===');
[YPredTest, probsTest] = classify(net, augimdsTest);
YTest = imdsTest.Labels;

accuracyTest = sum(YPredTest == YTest) / numel(YTest);
disp(['测试集准确率: ', num2str(accuracyTest * 100), '%']);

%% 8. 详细性能分析
% 混淆矩阵
figure('Position', [100, 100, 1200, 500]);

% 测试集混淆矩阵
subplot(1,2,1);
cmTest = confusionmat(YTest, YPredTest);
confusionchart(cmTest, categories(imdsTest.Labels));
title(['测试集混淆矩阵 (准确率: ', num2str(accuracyTest * 100, '%.2f'), '%)']);

% 验证集混淆矩阵
subplot(1,2,2);
cmVal = confusionmat(YValidation, YPredVal);
confusionchart(cmVal, categories(imdsValidation.Labels));
title(['验证集混淆矩阵 (准确率: ', num2str(accuracyVal * 100, '%.2f'), '%)']);

% 计算详细指标
YTestBinary = double(YTest == 'dogs');
YPredTestBinary = double(YPredTest == 'dogs');

TP = sum((YTestBinary == 1) & (YPredTestBinary == 1));
TN = sum((YTestBinary == 0) & (YPredTestBinary == 0));
FP = sum((YTestBinary == 0) & (YPredTestBinary == 1));
FN = sum((YTestBinary == 1) & (YPredTestBinary == 0));

precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1Score = 2 * (precision * recall) / (precision + recall);

disp('=== 详细性能指标 ===');
disp(['精确率 (Precision): ', num2str(precision * 100, '%.2f'), '%']);
disp(['召回率 (Recall): ', num2str(recall * 100, '%.2f'), '%']);
disp(['F1分数: ', num2str(f1Score * 100, '%.2f'), '%']);
disp(['真阳性 (TP): ', num2str(TP)]);
disp(['真阴性 (TN): ', num2str(TN)]);
disp(['假阳性 (FP): ', num2str(FP)]);
disp(['假阴性 (FN): ', num2str(FN)]);

%% 9. 可视化预测结果
% 随机选择一些测试图像进行可视化
numSamples = 16;
idx = randperm(numel(imdsTest.Files), min(numSamples, numel(imdsTest.Files)));

figure('Position', [100, 100, 1400, 800]);
for i = 1:length(idx)
    subplot(4, 4, i);
    
    % 读取并显示图像
    img = readimage(imdsTest, idx(i));
    imshow(img);
    
    % 获取真实标签和预测标签
    trueLabel = char(YTest(idx(i)));
    predLabel = char(YPredTest(idx(i)));
    
    % 获取预测概率
    predProb = max(probsTest(idx(i), :)) * 100;
    
    % 设置标题颜色
    if strcmp(trueLabel, predLabel)
        color = 'g';
    else
        color = 'r';
    end
    
    title(sprintf('真实: %s\n预测: %s (%.1f%%)', ...
        trueLabel, predLabel, predProb), ...
        'Color', color, 'FontSize', 9);
end
sgtitle('测试集预测结果示例（绿色:正确, 红色:错误）');

%% 10. 绘制ROC曲线
figure;
% 计算狗的预测概率
dogProbs = probsTest(:, 2);

% 计算ROC曲线
[X, Y, T, AUC] = perfcurve(YTestBinary, dogProbs, 1);

% 绘制ROC曲线
plot(X, Y, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
xlabel('假阳性率');
ylabel('真阳性率');
title(sprintf('ROC曲线 (AUC = %.3f)', AUC));
grid on;
legend(['模型 (AUC=', num2str(AUC, '%.3f'), ')'], '随机分类', 'Location', 'southeast');

%% 11. 训练过程可视化
figure('Position', [100, 100, 1200, 400]);

% 绘制训练和验证准确率
if isfield(trainInfo, 'TrainingAccuracy')
    subplot(1,2,1);
    plot(trainInfo.TrainingAccuracy, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(trainInfo.ValidationAccuracy, 'r-', 'LineWidth', 1.5);
    xlabel('迭代次数');
    ylabel('准确率 (%)');
    title('训练过程 - 准确率');
    legend('训练', '验证', 'Location', 'best');
    grid on;
    
    subplot(1,2,2);
    plot(trainInfo.TrainingLoss, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(trainInfo.ValidationLoss, 'r-', 'LineWidth', 1.5);
    xlabel('迭代次数');
    ylabel('损失');
    title('训练过程 - 损失');
    legend('训练损失', '验证损失', 'Location', 'best');
    grid on;
else
    % 如果训练信息中没有准确率，只绘制损失
    plot(trainInfo.TrainingLoss, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(trainInfo.ValidationLoss, 'r-', 'LineWidth', 1.5);
    xlabel('迭代次数');
    ylabel('损失');
    title('训练过程');
    legend('训练损失', '验证损失', 'Location', 'best');
    grid on;
end

%% 12. 保存实验结果
results = struct();
results.testAccuracy = accuracyTest;
results.valAccuracy = accuracyVal;
results.precision = precision;
results.recall = recall;
results.f1Score = f1Score;
results.AUC = AUC;
results.confusionMatrixTest = cmTest;
results.confusionMatrixVal = cmVal;
results.TP = TP;
results.TN = TN;
results.FP = FP;
results.FN = FN;

% 保存结果
save('experiment_results_fixed.mat', 'results', 'trainInfo');

% 生成实验报告
fid = fopen('experiment_summary_fixed.txt', 'w');
fprintf(fid, '猫狗分类实验报告（修复版）\n');
fprintf(fid, '===============================\n\n');
fprintf(fid, '数据集信息:\n');
fprintf(fid, '  训练集: %d 张图片\n', numel(imdsTrain.Files));
fprintf(fid, '  验证集: %d 张图片\n', numel(imdsValidation.Files));
fprintf(fid, '  测试集: %d 张图片\n\n', numel(imdsTest.Files));

fprintf(fid, '训练参数:\n');
fprintf(fid, '  图像尺寸: %dx%dx%d\n', imageSize(1), imageSize(2), imageSize(3));
fprintf(fid, '  批量大小: %d\n', miniBatchSize);
fprintf(fid, '  训练轮数: %d\n', maxEpochs);
fprintf(fid, '  初始学习率: %.4f\n\n', initialLearnRate);

fprintf(fid, '网络架构:\n');
fprintf(fid, '  卷积层: 4个\n');
fprintf(fid, '  全连接层: 3个\n');
fprintf(fid, '  总层数: %d\n\n', numel(layers));

fprintf(fid, '性能指标:\n');
fprintf(fid, '  测试集准确率: %.2f%%\n', accuracyTest * 100);
fprintf(fid, '  验证集准确率: %.2f%%\n', accuracyVal * 100);
fprintf(fid, '  精确率 (Precision): %.2f%%\n', precision * 100);
fprintf(fid, '  召回率 (Recall): %.2f%%\n', recall * 100);
fprintf(fid, '  F1分数: %.2f%%\n', f1Score * 100);
fprintf(fid, '  AUC: %.3f\n\n', AUC);

fprintf(fid, '混淆矩阵 (测试集):\n');
fprintf(fid, '          预测猫   预测狗\n');
fprintf(fid, '  实际猫   %4d     %4d\n', cmTest(1,1), cmTest(1,2));
fprintf(fid, '  实际狗   %4d     %4d\n', cmTest(2,1), cmTest(2,2));

fclose(fid);

disp('=== 实验完成 ===');
disp('所有结果已保存:');
disp('1. cat_dog_classifier_fixed.mat - 训练好的网络');
disp('2. experiment_results_fixed.mat - 实验数据');
disp('3. experiment_summary_fixed.txt - 实验总结');

%% 13. 可选的改进：使用多个模型的集成
disp('=== 可选：使用多个模型集成 ===');
choice = input('是否要训练多个模型进行集成？(y/n): ', 's');

if strcmpi(choice, 'y')
    disp('开始训练多个模型进行集成...');
    
    % 训练多个不同配置的模型
    numModels = 3;
    models = cell(1, numModels);
    accuracies = zeros(1, numModels);
    
    for modelIdx = 1:numModels
        fprintf('\n=== 训练模型 %d/%d ===\n', modelIdx, numModels);
        
        % 轻微修改网络结构
        if modelIdx == 1
            % 模型1：原始结构
            modelLayers = layers;
        elseif modelIdx == 2
            % 模型2：更多的卷积层
            modelLayers = [
                imageInputLayer(imageSize, 'Name', 'input', 'Normalization', 'zerocenter')
                convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
                convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
                batchNormalizationLayer('Name', 'bn2')
                reluLayer('Name', 'relu2')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
                convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
                batchNormalizationLayer('Name', 'bn3')
                reluLayer('Name', 'relu3')
                convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv4')
                batchNormalizationLayer('Name', 'bn4')
                reluLayer('Name', 'relu4')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
                fullyConnectedLayer(256, 'Name', 'fc1')
                batchNormalizationLayer('Name', 'bn5')
                reluLayer('Name', 'relu5')
                dropoutLayer(0.5, 'Name', 'dropout1')
                fullyConnectedLayer(2, 'Name', 'fc_output')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'classoutput')
            ];
        else
            % 模型3：更少的卷积层
            modelLayers = [
                imageInputLayer(imageSize, 'Name', 'input', 'Normalization', 'zerocenter')
                convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
                convolution2dLayer(5, 64, 'Padding', 'same', 'Name', 'conv2')
                batchNormalizationLayer('Name', 'bn2')
                reluLayer('Name', 'relu2')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
                fullyConnectedLayer(128, 'Name', 'fc1')
                batchNormalizationLayer('Name', 'bn3')
                reluLayer('Name', 'relu3')
                dropoutLayer(0.5, 'Name', 'dropout1')
                fullyConnectedLayer(2, 'Name', 'fc_output')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'classoutput')
            ];
        end
        
        % 训练当前模型
        modelOptions = trainingOptions('adam', ...
            'InitialLearnRate', initialLearnRate * (0.8 + 0.4*rand()), ...
            'MaxEpochs', 30, ...
            'MiniBatchSize', miniBatchSize, ...
            'ValidationData', augimdsValidation, ...
            'ValidationFrequency', validationFrequency, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', false, ...
            'Plots', 'none', ...
            'ExecutionEnvironment', 'auto', ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.5, ...
            'LearnRateDropPeriod', 8, ...
            'OutputNetwork', 'best-validation-loss');
        
        currentModel = trainNetwork(augimdsTrain, modelLayers, modelOptions);
        models{modelIdx} = currentModel;
        
        % 评估当前模型
        [YPredCurrent, ~] = classify(currentModel, augimdsTest);
        currentAccuracy = sum(YPredCurrent == YTest) / numel(YTest);
        accuracies(modelIdx) = currentAccuracy;
        
        fprintf('模型 %d 测试集准确率: %.2f%%\n', modelIdx, currentAccuracy * 100);
    end
    
    % 集成预测（投票法）
    allPredictions = zeros(numel(YTest), numModels);
    for i = 1:numModels
        [YPredModel, ~] = classify(models{i}, augimdsTest);
        allPredictions(:, i) = double(YPredModel == 'dogs');
    end
    
    % 多数投票
    ensemblePredictions = mode(allPredictions, 2);
    YPredEnsemble = categorical(ensemblePredictions, [0, 1], {'cats', 'dogs'});
    
    % 计算集成模型的准确率
    ensembleAccuracy = sum(YPredEnsemble == YTest) / numel(YTest);
    fprintf('\n集成模型测试集准确率: %.2f%%\n', ensembleAccuracy * 100);
    fprintf('相比最佳单个模型提升: %.2f%%\n', (ensembleAccuracy - max(accuracies)) * 100);
end