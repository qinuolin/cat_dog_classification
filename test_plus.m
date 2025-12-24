%% 猫狗分类器 - 修复版（完整功能）
clear; clc; close all;

%% 初始化程序
fprintf('===========================================\n');
fprintf('         猫狗分类器 v2.0\n');
fprintf('===========================================\n\n');

% 检查模型文件
if ~exist('cat_dog_classifier_fixed.mat', 'file')
    fprintf('❌ 错误: 模型文件不存在\n');
    fprintf('请确保 cat_dog_classifier_fixed.mat 在当前目录\n');
    fprintf('按任意键退出...\n');
    pause;
    return;
end

% 加载模型
fprintf('正在加载模型...\n');
try
    load('cat_dog_classifier_fixed.mat', 'net');
    fprintf('✅ 模型加载成功\n');
catch ME
    fprintf('❌ 加载模型失败: %s\n', ME.message);
    return;
end

% 获取模型信息
imageSize = [224, 224, 3];

% 修复：安全地获取类别名称
if isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork')
    if ~isempty(net.Layers)
        outputLayer = net.Layers(end);
        % 尝试获取类别信息
        if isprop(outputLayer, 'Classes')
            classesProp = outputLayer.Classes;
            if iscell(classesProp)
                classes = classesProp;
            elseif iscategorical(classesProp)
                classes = cellstr(classesProp);
            elseif isstring(classesProp)
                classes = cellstr(classesProp);
            elseif ischar(classesProp)
                classes = {classesProp};
            else
                classes = {'cat', 'dog'}; % 默认值
            end
        else
            classes = {'cat', 'dog'}; % 默认值
        end
    else
        classes = {'cat', 'dog'}; % 默认值
    end
else
    classes = {'cat', 'dog'}; % 默认值
end

% 确保classes是元胞数组且至少有两个元素
if ~iscell(classes)
    classes = {classes};
end

% 如果只有一个类别，添加另一个
if length(classes) < 2
    if contains(lower(char(classes{1})), 'cat')
        classes = {char(classes{1}), 'dog'};
    else
        classes = {'cat', char(classes{1})};
    end
end

fprintf('模型信息: 输入尺寸 %dx%dx%d\n', imageSize(1), imageSize(2), imageSize(3));
fprintf('类别: %s, %s\n\n', char(classes{1}), char(classes{2}));

% 全局配置
config.imageSize = imageSize;
config.classes = classes;
config.confidenceThreshold = 70; % 置信度阈值

%% 主菜单循环
while true
    fprintf('\n============ 主菜单 ============\n');
    fprintf('  1. 单张图片测试\n');
    fprintf('  2. 批量图片测试\n');
    fprintf('  3. 测试集性能评估\n');
    fprintf('  4. 摄像头实时识别\n');
    fprintf('  5. 查看模型信息\n');
    fprintf('  6. 帮助文档\n');
    fprintf('  0. 退出程序\n');
    fprintf('===============================\n');
    
    choice = input('请选择功能 (0-6): ', 's');
    
    switch choice
        case '1'
            testSingleImageUI(net, config);
        case '2'
            testBatchImagesUI(net, config);
        case '3'
            evaluatePerformance(net, config);
        case '4'
            testWithCamera(net, config);
        case '5'
            showModelInfo(net, config);
        case '6'
            showHelp();
        case '0'
            fprintf('\n感谢使用猫狗分类器！再见！\n');
            break;
        otherwise
            fprintf('❌ 无效选择，请重新输入\n');
    end
end

%% 功能1：单张图片测试（优化版）
function testSingleImageUI(net, config)
    fprintf('\n======== 单张图片测试 ========\n');
    fprintf('1. 选择图片文件\n');
    fprintf('2. 输入图片路径\n');
    fprintf('0. 返回主菜单\n');
    subChoice = input('请选择: ', 's');
    
    switch subChoice
        case '1'
            % 文件选择对话框
            [filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.gif', ...
                '图像文件 (*.jpg, *.jpeg, *.png, *.bmp, *.tif, *.gif)'}, ...
                '选择要分类的图片');
            
            if isequal(filename, 0)
                fprintf('已取消选择\n');
                return;
            end
            
            imgPath = fullfile(pathname, filename);
            
        case '2'
            % 手动输入路径
            imgPath = input('请输入图片完整路径: ', 's');
            
        case '0'
            return;
            
        otherwise
            fprintf('无效选择\n');
            return;
    end
    
    % 检查文件是否存在
    if ~exist(imgPath, 'file')
        fprintf('❌ 错误: 文件不存在\n');
        return;
    end
    
    % 执行测试
    fprintf('\n正在处理图片...\n');
    [result, success] = processSingleImage(net, imgPath, config);
    
    if success
        showSingleImageResult(result, config);
    else
        fprintf('❌ 处理图片失败\n');
    end
end

%% 功能2：批量图片测试（优化版）
function testBatchImagesUI(net, config)
    fprintf('\n======== 批量图片测试 ========\n');
    fprintf('1. 选择图片文件夹\n');
    fprintf('2. 输入文件夹路径\n');
    fprintf('0. 返回主菜单\n');
    subChoice = input('请选择: ', 's');
    
    switch subChoice
        case '1'
            % 文件夹选择对话框
            folderPath = uigetdir(pwd, '选择包含图片的文件夹');
            if isequal(folderPath, 0)
                fprintf('已取消选择\n');
                return;
            end
            
        case '2'
            % 手动输入路径
            folderPath = input('请输入文件夹完整路径: ', 's');
            
        case '0'
            return;
            
        otherwise
            fprintf('无效选择\n');
            return;
    end
    
    % 检查文件夹是否存在
    if ~exist(folderPath, 'dir')
        fprintf('❌ 错误: 文件夹不存在\n');
        return;
    end
    
    % 获取所有图片文件
    fprintf('\n正在搜索图片文件...\n');
    imageFiles = getImageFiles(folderPath);
    
    if isempty(imageFiles)
        fprintf('❌ 未找到图片文件\n');
        return;
    end
    
    fprintf('找到 %d 个图片文件\n', length(imageFiles));
    
    % 选择处理方式
    fprintf('\n处理选项:\n');
    fprintf('1. 处理所有图片 (%d 个)\n', length(imageFiles));
    fprintf('2. 处理前N个图片\n');
    fprintf('3. 随机抽样处理\n');
    fprintf('0. 返回\n');
    
    processChoice = input('请选择: ', 's');
    
    switch processChoice
        case '1'
            % 处理所有图片
            indices = 1:length(imageFiles);
            numToProcess = length(imageFiles);
            
        case '2'
            % 处理前N个
            numToProcess = input('请输入要处理的图片数量: ');
            if numToProcess <= 0 || numToProcess > length(imageFiles)
                fprintf('❌ 无效数量\n');
                return;
            end
            indices = 1:numToProcess;
            
        case '3'
            % 随机抽样
            numToProcess = input('请输入要处理的图片数量: ');
            if numToProcess <= 0 || numToProcess > length(imageFiles)
                fprintf('❌ 无效数量\n');
                return;
            end
            indices = randperm(length(imageFiles), numToProcess);
            
        case '0'
            return;
            
        otherwise
            fprintf('❌ 无效选择\n');
            return;
    end
    
    % 执行批量处理
    fprintf('\n开始批量处理...\n');
    [results, success] = processBatchImages(net, imageFiles, indices, config, folderPath);
    
    if success
        showBatchResults(results, config);
    else
        fprintf('❌ 批量处理失败\n');
    end
end

%% 功能3：测试集性能评估（简化版）
function evaluatePerformance(net, config)
    fprintf('\n======== 测试集性能评估 ========\n');
    
    % 询问测试集路径
    testSetPath = input('请输入测试集文件夹路径 (按Enter使用默认路径 D:\深度学习\test): ', 's');
    if isempty(testSetPath)
        testSetPath = 'D:\深度学习\test';
    end
    
    % 检查文件夹是否存在
    if ~exist(testSetPath, 'dir')
        fprintf('❌ 错误: 测试集文件夹不存在\n');
        return;
    end
    
    % 检查是否有子文件夹（类别）
    subfolders = dir(testSetPath);
    subfolders = subfolders([subfolders.isdir]);
    subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));
    
    if isempty(subfolders)
        fprintf('⚠️ 测试集没有子文件夹，无法进行准确率评估\n');
        fprintf('只能进行预测统计\n');
        mode = 'predict_only';
    else
        fprintf('发现 %d 个子文件夹，将进行准确率评估\n', length(subfolders));
        mode = 'evaluate';
    end
    
    % 收集所有图片
    fprintf('\n正在收集测试图片...\n');
    allImageFiles = {};
    allTrueLabels = {};
    
    if strcmp(mode, 'evaluate')
        % 从子文件夹收集图片
        for i = 1:length(subfolders)
            subfolderPath = fullfile(testSetPath, subfolders(i).name);
            imageFiles = getImageFiles(subfolderPath);
            
            for j = 1:length(imageFiles)
                allImageFiles{end+1} = fullfile(subfolderPath, imageFiles(j).name);
                allTrueLabels{end+1} = subfolders(i).name;
            end
        end
    else
        % 从根目录收集图片
        imageFiles = getImageFiles(testSetPath);
        for j = 1:length(imageFiles)
            allImageFiles{end+1} = fullfile(testSetPath, imageFiles(j).name);
            allTrueLabels{end+1} = 'unknown';
        end
    end
    
    if isempty(allImageFiles)
        fprintf('❌ 未找到测试图片\n');
        return;
    end
    
    fprintf('找到 %d 个测试图片\n', length(allImageFiles));
    
    % 询问评估样本数
    if length(allImageFiles) > 100
        fprintf('\n测试图片较多，建议抽样测试\n');
        sampleSize = input('请输入测试样本数（按Enter使用全部）: ', 's');
        if ~isempty(sampleSize)
            sampleSize = str2double(sampleSize);
            if ~isnan(sampleSize) && sampleSize > 0 && sampleSize <= length(allImageFiles)
                indices = randperm(length(allImageFiles), sampleSize);
                fprintf('将随机测试 %d 个样本\n', sampleSize);
            else
                indices = 1:length(allImageFiles);
                fprintf('将测试全部 %d 个样本\n', length(allImageFiles));
            end
        else
            indices = 1:length(allImageFiles);
            fprintf('将测试全部 %d 个样本\n', length(allImageFiles));
        end
    else
        indices = 1:length(allImageFiles);
        fprintf('将测试全部 %d 个样本\n', length(allImageFiles));
    end
    
    % 执行评估
    fprintf('\n开始性能评估...\n');
    [evaluationResults, success] = evaluateModel(net, allImageFiles, allTrueLabels, indices, config, mode);
    
    if success
        showEvaluationResults(evaluationResults, config);
    else
        fprintf('❌ 评估失败\n');
    end
end

%% 功能4：摄像头实时识别（简化版）
function testWithCamera(net, config)
    fprintf('\n======== 摄像头实时识别 ========\n');
    
    % 检查摄像头支持
    if ~ispc
        fprintf('❌ 此功能仅支持Windows系统\n');
        return;
    end
    
    try
        % 检查Image Acquisition Toolbox是否可用
        if ~license('test', 'Image_Acquisition_Toolbox')
            fprintf('❌ 需要Image Acquisition Toolbox\n');
            return;
        end
        
        info = imaqhwinfo;
        if isempty(info.InstalledAdaptors)
            fprintf('❌ 未检测到摄像头\n');
            return;
        end
        
        fprintf('检测到摄像头适配器: %s\n', strjoin(info.InstalledAdaptors, ', '));
        fprintf('\n1. 简单摄像头测试\n');
        fprintf('0. 返回\n');
        
        camChoice = input('请选择: ', 's');
        
        if strcmp(camChoice, '1')
            simpleCameraTest(net, config);
        elseif strcmp(camChoice, '0')
            return;
        else
            fprintf('❌ 无效选择\n');
        end
        
    catch ME
        fprintf('❌ 摄像头错误: %s\n', ME.message);
    end
end

%% 功能5：查看模型信息
function showModelInfo(net, config)
    fprintf('\n======== 模型信息 ========\n');
    
    % 显示基础信息
    fprintf('模型类型: %s\n', class(net));
    fprintf('输入尺寸: %dx%dx%d\n', config.imageSize(1), config.imageSize(2), config.imageSize(3));
    
    if iscell(config.classes) && length(config.classes) >= 2
        fprintf('输出类别: %s, %s\n', char(config.classes{1}), char(config.classes{2}));
    else
        fprintf('输出类别: 未知\n');
    end
    
    % 显示层信息
    if isprop(net, 'Layers')
        numLayers = numel(net.Layers);
        fprintf('网络层数: %d\n', numLayers);
        
        % 显示前5层和后5层的信息
        fprintf('\n网络层结构:\n');
        fprintf('%-5s %-25s %-20s\n', '编号', '层类型', '层名称');
        fprintf('%s\n', repmat('-', 50, 1));
        
        for i = 1:min(5, numLayers)
            layer = net.Layers(i);
            fprintf('%-5d %-25s %-20s\n', i, class(layer), layer.Name);
        end
        
        if numLayers > 10
            fprintf('... (省略中间 %d 层) ...\n', numLayers - 10);
        end
        
        for i = max(1, numLayers-4):numLayers
            layer = net.Layers(i);
            fprintf('%-5d %-25s %-20s\n', i, class(layer), layer.Name);
        end
    end
    
    fprintf('\n性能指标 (根据训练结果):\n');
    fprintf('  测试集准确率: 93.2%%\n');
    fprintf('  验证集准确率: 92.9%%\n');
    fprintf('  AUC值: 0.982\n');
    
    fprintf('\n按任意键继续...\n');
    pause;
end

%% 功能6：帮助文档
function showHelp()
    fprintf('\n======== 帮助文档 ========\n');
    fprintf('\n猫狗分类器使用说明\n');
    fprintf('版本: 2.0\n');
    fprintf('模型准确率: 93.2%%\n');
    
    fprintf('\n功能说明:\n');
    fprintf('1. 单张图片测试 - 测试单张猫或狗图片\n');
    fprintf('2. 批量图片测试 - 测试整个文件夹的图片\n');
    fprintf('3. 测试集性能评估 - 评估模型在测试集上的性能\n');
    fprintf('4. 摄像头实时识别 - 使用摄像头进行实时识别\n');
    fprintf('5. 查看模型信息 - 显示模型详细信息\n');
    fprintf('6. 帮助文档 - 显示本帮助信息\n');
    fprintf('0. 退出程序 - 退出应用程序\n');
    
    fprintf('\n使用技巧:\n');
    fprintf('• 支持格式: JPG, JPEG, PNG, BMP, TIF, GIF\n');
    fprintf('• 建议图片尺寸: 大于224x224像素\n');
    fprintf('• 高置信度: >90%% 结果可靠\n');
    fprintf('• 中等置信度: 70%%-90%% 结果基本可靠\n');
    fprintf('• 低置信度: <70%% 模型不太确定\n');
    
    fprintf('\n常见问题:\n');
    fprintf('1. 模型文件不存在 - 确保 cat_dog_classifier_fixed.mat 在当前目录\n');
    fprintf('2. 图片加载失败 - 检查图片格式和路径是否正确\n');
    fprintf('3. 摄像头无法使用 - 检查摄像头连接\n');
    
    fprintf('\n按任意键继续...\n');
    pause;
end

%% ========== 核心处理函数 ==========

%% 处理单张图片
function [result, success] = processSingleImage(net, imgPath, config)
    result = struct();
    success = false;
    
    try
        tic;
        
        % 读取图片
        img = imread(imgPath);
        
        % 获取文件信息
        [filepath, filename, ext] = fileparts(imgPath);
        result.filename = [filename, ext];
        result.fullpath = imgPath;
        result.originalSize = size(img);
        
        % 预处理图片
        imgProcessed = preprocessImage(img, config.imageSize);
        
        % 进行预测
        [predLabel, predProbs] = classify(net, imgProcessed);
        
        % 处理结果
        result.prediction = char(predLabel);
        result.confidence = max(predProbs) * 100;
        result.probs = predProbs * 100;
        result.processingTime = toc;
        result.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        
        % 转换为中文标签
        if contains(result.prediction, 'cat', 'IgnoreCase', true)
            result.predictionChinese = '猫';
        elseif contains(result.prediction, 'dog', 'IgnoreCase', true)
            result.predictionChinese = '狗';
        else
            result.predictionChinese = result.prediction;
        end
        
        success = true;
        
    catch ME
        fprintf('处理图片时出错: %s\n', ME.message);
        result.error = ME.message;
    end
end

%% 显示单张图片结果
function showSingleImageResult(result, config)
    fprintf('\n======== 预测结果 ========\n');
    fprintf('文件名: %s\n', result.filename);
    fprintf('预测结果: %s (%s)\n', result.predictionChinese, result.prediction);
    fprintf('置信度: %.1f%%\n', result.confidence);
    fprintf('处理时间: %.3f秒\n', result.processingTime);
    fprintf('=========================\n');
    
    % 读取并显示图片
    try
        img = imread(result.fullpath);
        
        % 创建结果显示图
        fig = figure('Name', '预测结果', 'NumberTitle', 'off', ...
            'Position', [100, 100, 800, 400]);
        
        % 原始图片
        subplot(1, 2, 1);
        imshow(img);
        title('原始图片', 'FontSize', 12, 'FontWeight', 'bold');
        xlabel(sprintf('尺寸: %dx%d', size(img,1), size(img,2)), 'FontSize', 10);
        
        % 结果展示
        subplot(1, 2, 2);
        axis off;
        
        % 设置文本位置
        textY = 0.9;
        
        % 标题
        text(0.5, textY, '预测结果详情', 'FontSize', 16, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'Units', 'normalized');
        textY = textY - 0.1;
        
        % 文件名
        text(0.1, textY, sprintf('文件名: %s', result.filename), ...
            'FontSize', 12, 'Units', 'normalized');
        textY = textY - 0.08;
        
        % 预测结果（根据置信度显示不同颜色）
        if result.confidence >= 90
            confColor = [0, 0.7, 0]; % 绿色
            confText = '✓ 高置信度';
        elseif result.confidence >= 70
            confColor = [0.9, 0.6, 0]; % 橙色
            confText = '⚠ 中等置信度';
        else
            confColor = [0.9, 0, 0]; % 红色
            confText = '⚠ 低置信度';
        end
        
        if strcmp(result.predictionChinese, '猫')
            predColor = [0.9, 0.2, 0.2]; % 红色
        else
            predColor = [0.2, 0.2, 0.9]; % 蓝色
        end
        
        text(0.1, textY, sprintf('预测结果: %s', result.predictionChinese), ...
            'FontSize', 14, 'FontWeight', 'bold', 'Color', predColor, 'Units', 'normalized');
        textY = textY - 0.08;
        
        text(0.1, textY, sprintf('置信度: %.1f%% (%s)', result.confidence, confText), ...
            'FontSize', 14, 'FontWeight', 'bold', 'Color', confColor, 'Units', 'normalized');
        textY = textY - 0.08;
        
        % 概率信息
        if length(result.probs) >= 2
            text(0.1, textY, sprintf('猫的概率: %.1f%%', result.probs(1)), ...
                'FontSize', 12, 'Color', [0.9, 0.2, 0.2], 'Units', 'normalized');
            textY = textY - 0.06;
            
            text(0.1, textY, sprintf('狗的概率: %.1f%%', result.probs(2)), ...
                'FontSize', 12, 'Color', [0.2, 0.2, 0.9], 'Units', 'normalized');
            textY = textY - 0.06;
        end
        
        % 处理时间
        text(0.1, textY, sprintf('处理时间: %.3f秒', result.processingTime), ...
            'FontSize', 12, 'Units', 'normalized');
        
        % 添加概率条
        if length(result.probs) >= 2
            axes('Position', [0.3, 0.1, 0.4, 0.2]);
            hold on;
            
            barWidth = 0.6;
            barHeight = 0.3;
            
            % 猫概率条
            rectangle('Position', [0.2, 0.1, barWidth * (result.probs(1)/100), barHeight], ...
                'FaceColor', [0.9, 0.2, 0.2], 'EdgeColor', 'none');
            
            % 狗概率条
            rectangle('Position', [0.2 + barWidth * (result.probs(1)/100), 0.1, barWidth * (result.probs(2)/100), barHeight], ...
                'FaceColor', [0.2, 0.2, 0.9], 'EdgeColor', 'none');
            
            % 外框
            rectangle('Position', [0.2, 0.1, barWidth, barHeight], ...
                'EdgeColor', 'black', 'LineWidth', 1);
            
            axis off;
            text(0.2, 0.45, '猫', 'Color', [0.9, 0.2, 0.2], 'FontWeight', 'bold');
            text(0.2 + barWidth - 0.05, 0.45, '狗', 'Color', [0.2, 0.2, 0.9], 'FontWeight', 'bold');
        end
        
        % 询问是否保存结果
        saveChoice = input('\n是否保存预测结果？(y/n): ', 's');
        if strcmpi(saveChoice, 'y') || strcmpi(saveChoice, '是')
            saveSingleResult(result);
        end
        
    catch ME
        fprintf('显示结果时出错: %s\n', ME.message);
    end
end

%% 保存单张结果
function saveSingleResult(result)
    % 创建结果文件夹
    if ~exist('results', 'dir')
        mkdir('results');
    end
    
    % 生成文件名
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename = sprintf('prediction_%s_%s', result.predictionChinese, timestamp);
    
    % 保存为MAT文件
    matPath = fullfile('results', [filename, '.mat']);
    save(matPath, 'result');
    fprintf('✅ 结果已保存: %s\n', matPath);
    
    % 保存为文本文件
    txtPath = fullfile('results', [filename, '.txt']);
    fid = fopen(txtPath, 'w');
    fprintf(fid, '猫狗分类预测结果\n');
    fprintf(fid, '====================\n\n');
    fprintf(fid, '文件名: %s\n', result.filename);
    fprintf(fid, '预测结果: %s (%s)\n', result.predictionChinese, result.prediction);
    fprintf(fid, '置信度: %.2f%%\n', result.confidence);
    fprintf(fid, '处理时间: %.3f秒\n', result.processingTime);
    fprintf(fid, '测试时间: %s\n', result.timestamp);
    fclose(fid);
    fprintf('✅ 文本结果已保存: %s\n', txtPath);
end

%% 批量处理图片
function [results, success] = processBatchImages(net, imageFiles, indices, config, folderPath)
    results = struct();
    success = false;
    
    numFiles = length(indices);
    
    % 初始化结果数组
    results.filenames = cell(numFiles, 1);
    results.predictions = cell(numFiles, 1);
    results.predictionsChinese = cell(numFiles, 1);
    results.confidences = zeros(numFiles, 1);
    results.processingTimes = zeros(numFiles, 1);
    results.success = false(numFiles, 1);
    
    % 显示进度
    fprintf('\n');
    fprintf('进度: [');
    progressWidth = 50;
    
    % 处理每张图片
    for i = 1:numFiles
        idx = indices(i);
        imgPath = fullfile(folderPath, imageFiles(idx).name);
        
        try
            % 处理单张图片
            tic;
            img = imread(imgPath);
            imgProcessed = preprocessImage(img, config.imageSize);
            [predLabel, predProbs] = classify(net, imgProcessed);
            
            % 保存结果
            results.filenames{i} = imageFiles(idx).name;
            results.predictions{i} = char(predLabel);
            results.confidences(i) = max(predProbs) * 100;
            results.processingTimes(i) = toc;
            results.success(i) = true;
            
            % 转换为中文标签
            if contains(results.predictions{i}, 'cat', 'IgnoreCase', true)
                results.predictionsChinese{i} = '猫';
            elseif contains(results.predictions{i}, 'dog', 'IgnoreCase', true)
                results.predictionsChinese{i} = '狗';
            else
                results.predictionsChinese{i} = results.predictions{i};
            end
            
        catch ME
            % 记录错误
            results.filenames{i} = imageFiles(idx).name;
            results.predictions{i} = '处理失败';
            results.predictionsChinese{i} = '处理失败';
            results.confidences(i) = 0;
            results.processingTimes(i) = 0;
            results.success(i) = false;
        end
        
        % 更新进度
        progress = floor(i / numFiles * progressWidth);
        fprintf('\r进度: [%s%s] %d/%d', ...
            repmat('=', 1, progress), ...
            repmat(' ', 1, progressWidth - progress), ...
            i, numFiles);
    end
    
    fprintf(']\n');
    
    % 统计信息
    results.numTotal = numFiles;
    results.numSuccess = sum(results.success);
    results.numFailed = numFiles - results.numSuccess;
    
    if results.numSuccess > 0
        results.avgConfidence = mean(results.confidences(results.success));
        results.avgTime = mean(results.processingTimes(results.success));
    else
        results.avgConfidence = 0;
        results.avgTime = 0;
    end
    
    % 类别统计
    catIndices = strcmp(results.predictionsChinese, '猫') & results.success;
    dogIndices = strcmp(results.predictionsChinese, '狗') & results.success;
    results.numCats = sum(catIndices);
    results.numDogs = sum(dogIndices);
    
    success = true;
end

%% 显示批量结果
function showBatchResults(results, config)
    fprintf('\n======== 批量处理结果 ========\n');
    fprintf('总处理数: %d\n', results.numTotal);
    fprintf('成功处理: %d (%.1f%%)\n', results.numSuccess, results.numSuccess/results.numTotal*100);
    fprintf('处理失败: %d (%.1f%%)\n', results.numFailed, results.numFailed/results.numTotal*100);
    
    if results.numSuccess > 0
        fprintf('猫的数量: %d (%.1f%%)\n', results.numCats, results.numCats/results.numSuccess*100);
        fprintf('狗的数量: %d (%.1f%%)\n', results.numDogs, results.numDogs/results.numSuccess*100);
        fprintf('平均置信度: %.1f%%\n', results.avgConfidence);
        fprintf('平均处理时间: %.3f秒\n', results.avgTime);
    end
    
    fprintf('=============================\n');
    
    % 显示前10个结果
    if results.numTotal > 0
        fprintf('\n前10个结果:\n');
        fprintf('%-25s %-6s %-10s\n', '文件名', '预测', '置信度');
        fprintf('%s\n', repmat('-', 45, 1));
        
        numToShow = min(10, results.numTotal);
        for i = 1:numToShow
            filename = results.filenames{i};
            if length(filename) > 22
                filename = [filename(1:19), '...'];
            end
            
            fprintf('%-25s %-6s %-10.1f\n', ...
                filename, results.predictionsChinese{i}, results.confidences(i));
        end
    end
    
    % 询问是否保存结果
    saveChoice = input('\n是否保存批量处理结果？(y/n): ', 's');
    if strcmpi(saveChoice, 'y') || strcmpi(saveChoice, '是')
        saveBatchResults(results);
    end
end

%% 保存批量结果
function saveBatchResults(results)
    % 创建结果文件夹
    if ~exist('batch_results', 'dir')
        mkdir('batch_results');
    end
    
    % 生成时间戳
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    
    % 保存为MAT文件
    matPath = fullfile('batch_results', sprintf('batch_results_%s.mat', timestamp));
    save(matPath, 'results');
    fprintf('✅ 批量结果已保存: %s\n', matPath);
    
    % 生成文本报告
    txtPath = fullfile('batch_results', sprintf('batch_report_%s.txt', timestamp));
    fid = fopen(txtPath, 'w');
    
    fprintf(fid, '猫狗分类批量测试报告\n');
    fprintf(fid, '生成时间: %s\n\n', timestamp);
    
    fprintf(fid, '统计摘要:\n');
    fprintf(fid, '  总处理数: %d\n', results.numTotal);
    fprintf(fid, '  成功处理: %d (%.1f%%)\n', results.numSuccess, results.numSuccess/results.numTotal*100);
    fprintf(fid, '  处理失败: %d (%.1f%%)\n', results.numFailed, results.numFailed/results.numTotal*100);
    
    if results.numSuccess > 0
        fprintf(fid, '  猫的数量: %d (%.1f%%)\n', results.numCats, results.numCats/results.numSuccess*100);
        fprintf(fid, '  狗的数量: %d (%.1f%%)\n', results.numDogs, results.numDogs/results.numSuccess*100);
        fprintf(fid, '  平均置信度: %.1f%%\n', results.avgConfidence);
        fprintf(fid, '  平均处理时间: %.3f秒\n\n', results.avgTime);
    end
    
    fprintf(fid, '详细结果:\n');
    fprintf(fid, '%-30s %-8s %-10s\n', '文件名', '预测', '置信度');
    fprintf(fid, '%s\n', repmat('-', 50, 1));
    
    for i = 1:min(100, results.numTotal)
        if results.success(i)
            fprintf(fid, '%-30s %-8s %-10.1f\n', ...
                results.filenames{i}(1:min(30, length(results.filenames{i}))), ...
                results.predictionsChinese{i}, ...
                results.confidences(i));
        else
            fprintf(fid, '%-30s %-8s %-10s\n', ...
                results.filenames{i}(1:min(30, length(results.filenames{i}))), ...
                '失败', '0.0');
        end
    end
    
    fclose(fid);
    fprintf('✅ 文本报告已保存: %s\n', txtPath);
end

%% 模型评估
function [evaluationResults, success] = evaluateModel(net, allImageFiles, allTrueLabels, indices, config, mode)
    evaluationResults = struct();
    success = false;
    
    numFiles = length(indices);
    
    % 初始化结果
    results = struct();
    results.filenames = cell(numFiles, 1);
    results.predictions = cell(numFiles, 1);
    results.predictionsChinese = cell(numFiles, 1);
    results.confidences = zeros(numFiles, 1);
    results.success = false(numFiles, 1);
    
    if strcmp(mode, 'evaluate')
        results.trueLabels = cell(numFiles, 1);
        results.trueLabelsChinese = cell(numFiles, 1);
    end
    
    % 处理图片
    fprintf('\n正在评估模型...\n');
    
    for i = 1:numFiles
        idx = indices(i);
        imgPath = allImageFiles{idx};
        
        try
            % 处理单张图片
            img = imread(imgPath);
            imgProcessed = preprocessImage(img, config.imageSize);
            [predLabel, predProbs] = classify(net, imgProcessed);
            
            % 保存结果
            [~, filename, ext] = fileparts(imgPath);
            results.filenames{i} = [filename, ext];
            results.predictions{i} = char(predLabel);
            results.confidences(i) = max(predProbs) * 100;
            results.success(i) = true;
            
            % 转换为中文标签
            if contains(results.predictions{i}, 'cat', 'IgnoreCase', true)
                results.predictionsChinese{i} = '猫';
            elseif contains(results.predictions{i}, 'dog', 'IgnoreCase', true)
                results.predictionsChinese{i} = '狗';
            else
                results.predictionsChinese{i} = results.predictions{i};
            end
            
            % 保存真实标签
            if strcmp(mode, 'evaluate')
                results.trueLabels{i} = allTrueLabels{idx};
                if contains(results.trueLabels{i}, 'cat', 'IgnoreCase', true)
                    results.trueLabelsChinese{i} = '猫';
                elseif contains(results.trueLabels{i}, 'dog', 'IgnoreCase', true)
                    results.trueLabelsChinese{i} = '狗';
                else
                    results.trueLabelsChinese{i} = results.trueLabels{i};
                end
            end
            
        catch ME
            results.filenames{i} = allImageFiles{idx};
            results.predictions{i} = '处理失败';
            results.predictionsChinese{i} = '处理失败';
            results.confidences(i) = 0;
            results.success(i) = false;
            
            if strcmp(mode, 'evaluate')
                results.trueLabels{i} = allTrueLabels{idx};
            end
        end
        
        % 显示进度
        if mod(i, 10) == 0 || i == numFiles
            fprintf('  已处理: %d/%d\n', i, numFiles);
        end
    end
    
    % 计算评估指标
    evaluationResults.results = results;
    evaluationResults.numTotal = numFiles;
    evaluationResults.numSuccess = sum(results.success);
    evaluationResults.numFailed = numFiles - evaluationResults.numSuccess;
    
    if strcmp(mode, 'evaluate')
        % 计算准确率
        correct = 0;
        catAsDog = 0;
        dogAsCat = 0;
        catTotal = 0;
        dogTotal = 0;
        
        for i = 1:numFiles
            if results.success(i)
                trueLabel = results.trueLabelsChinese{i};
                predLabel = results.predictionsChinese{i};
                
                if strcmp(trueLabel, '猫')
                    catTotal = catTotal + 1;
                elseif strcmp(trueLabel, '狗')
                    dogTotal = dogTotal + 1;
                end
                
                if strcmp(trueLabel, predLabel)
                    correct = correct + 1;
                elseif strcmp(trueLabel, '猫') && strcmp(predLabel, '狗')
                    catAsDog = catAsDog + 1;
                elseif strcmp(trueLabel, '狗') && strcmp(predLabel, '猫')
                    dogAsCat = dogAsCat + 1;
                end
            end
        end
        
        totalEvaluated = catTotal + dogTotal;
        
        if totalEvaluated > 0
            evaluationResults.accuracy = correct / totalEvaluated * 100;
            evaluationResults.numEvaluated = totalEvaluated;
            evaluationResults.numCorrect = correct;
            evaluationResults.catAsDog = catAsDog;
            evaluationResults.dogAsCat = dogAsCat;
            evaluationResults.catTotal = catTotal;
            evaluationResults.dogTotal = dogTotal;
        else
            evaluationResults.accuracy = 0;
            evaluationResults.numEvaluated = 0;
        end
    else
        % 仅预测模式
        catIndices = strcmp(results.predictionsChinese, '猫') & results.success;
        dogIndices = strcmp(results.predictionsChinese, '狗') & results.success;
        evaluationResults.numCats = sum(catIndices);
        evaluationResults.numDogs = sum(dogIndices);
    end
    
    success = true;
end

%% 显示评估结果
function showEvaluationResults(evaluationResults, config)
    fprintf('\n======== 模型评估结果 ========\n');
    fprintf('测试样本数: %d\n', evaluationResults.numTotal);
    fprintf('成功处理: %d (%.1f%%)\n', ...
        evaluationResults.numSuccess, evaluationResults.numSuccess/evaluationResults.numTotal*100);
    
    if isfield(evaluationResults, 'accuracy')
        % 完整评估模式
        fprintf('有效评估样本: %d\n', evaluationResults.numEvaluated);
        if evaluationResults.numEvaluated > 0
            fprintf('正确预测: %d (%.1f%%)\n', ...
                evaluationResults.numCorrect, evaluationResults.accuracy);
            fprintf('猫被误判为狗: %d\n', evaluationResults.catAsDog);
            fprintf('狗被误判为猫: %d\n', evaluationResults.dogAsCat);
            
            % 显示混淆矩阵
            fprintf('\n混淆矩阵:\n');
            fprintf('          预测猫   预测狗\n');
            
            trueCatCorrect = evaluationResults.catTotal - evaluationResults.catAsDog;
            trueDogCorrect = evaluationResults.dogTotal - evaluationResults.dogAsCat;
            
            fprintf('  实际猫   %4d     %4d\n', trueCatCorrect, evaluationResults.catAsDog);
            fprintf('  实际狗   %4d     %4d\n', evaluationResults.dogAsCat, trueDogCorrect);
        end
    else
        % 仅预测模式
        fprintf('预测为猫: %d (%.1f%%)\n', ...
            evaluationResults.numCats, evaluationResults.numCats/evaluationResults.numSuccess*100);
        fprintf('预测为狗: %d (%.1f%%)\n', ...
            evaluationResults.numDogs, evaluationResults.numDogs/evaluationResults.numSuccess*100);
    end
    
    fprintf('===============================\n');
    
    % 保存评估结果
    saveChoice = input('\n是否保存评估结果？(y/n): ', 's');
    if strcmpi(saveChoice, 'y') || strcmpi(saveChoice, '是')
        timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
        
        if ~exist('evaluation_results', 'dir')
            mkdir('evaluation_results');
        end
        
        matPath = fullfile('evaluation_results', sprintf('evaluation_%s.mat', timestamp));
        save(matPath, 'evaluationResults');
        fprintf('✅ 评估结果已保存: %s\n', matPath);
    end
end

%% 简单摄像头测试
function simpleCameraTest(net, config)
    fprintf('\n正在启动摄像头...\n');
    fprintf('按任意键停止\n');
    
    try
        % 创建视频输入对象（使用默认摄像头）
        vid = videoinput('winvideo', 1);
        
        % 预览
        preview(vid);
        
        % 创建显示窗口
        fig = figure('Name', '摄像头测试', 'NumberTitle', 'off', ...
            'Position', [100, 100, 800, 600]);
        
        fprintf('摄像头已启动，按任意键拍照识别\n');
        
        while ishandle(fig)
            % 等待按键
            waitforbuttonpress;
            
            % 获取一帧
            frame = getsnapshot(vid);
            
            % 预处理
            imgProcessed = preprocessImage(frame, config.imageSize);
            
            % 预测
            [predLabel, predProbs] = classify(net, imgProcessed);
            confidence = max(predProbs) * 100;
            
            % 转换为中文标签
            predStr = char(predLabel);
            if contains(predStr, 'cat', 'IgnoreCase', true)
                predChinese = '猫';
                color = [0.9, 0.2, 0.2];
            elseif contains(predStr, 'dog', 'IgnoreCase', true)
                predChinese = '狗';
                color = [0.2, 0.2, 0.9];
            else
                predChinese = predStr;
                color = [0.2, 0.2, 0.2];
            end
            
            % 显示结果
            clf;
            subplot(1, 2, 1);
            imshow(frame);
            title('摄像头画面', 'FontSize', 12);
            
            subplot(1, 2, 2);
            imshow(imgProcessed);
            title(sprintf('预测: %s (%.1f%%)', predChinese, confidence), ...
                'FontSize', 14, 'FontWeight', 'bold', 'Color', color);
            
            fprintf('识别结果: %s (%.1f%%)\n', predChinese, confidence);
            
            % 询问是否保存
            saveChoice = input('是否保存这张图片？(y/n): ', 's');
            if strcmpi(saveChoice, 'y') || strcmpi(saveChoice, '是')
                timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
                filename = sprintf('capture_%s_%s.jpg', predChinese, timestamp);
                
                if ~exist('captures', 'dir')
                    mkdir('captures');
                end
                
                imwrite(frame, fullfile('captures', filename));
                fprintf('图片已保存: captures/%s\n', filename);
            end
            
            fprintf('\n按任意键继续拍照，或关闭窗口退出\n');
        end
        
        % 清理
        closepreview(vid);
        delete(vid);
        clear vid;
        
    catch ME
        fprintf('摄像头错误: %s\n', ME.message);
    end
end

%% ========== 辅助函数 ==========

%% 图片预处理
function imgProcessed = preprocessImage(img, targetSize)
    % 调整大小
    imgResized = imresize(img, targetSize(1:2));
    
    % 确保是RGB图像
    if size(imgResized, 3) == 1
        % 灰度图转RGB
        imgProcessed = repmat(imgResized, [1, 1, 3]);
    elseif size(imgResized, 3) == 4
        % RGBA转RGB
        imgProcessed = imgResized(:, :, 1:3);
    else
        imgProcessed = imgResized;
    end
end

%% 获取文件夹中所有图片文件
function imageFiles = getImageFiles(folderPath)
    % 支持的图片格式
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'};
    
    % 获取所有文件
    allFiles = dir(folderPath);
    
    % 筛选图片文件
    imageFiles = [];
    for i = 1:length(allFiles)
        if ~allFiles(i).isdir
            [~, ~, ext] = fileparts(allFiles(i).name);
            if any(strcmpi(ext, extensions))
                imageFiles = [imageFiles; allFiles(i)];
            end
        end
    end
end