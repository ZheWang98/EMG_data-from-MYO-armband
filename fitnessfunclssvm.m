function fitness = fitnessfunclssvm(x, train_EMG, train_EMG_labels)
% %% 定义适应度函数
% %p tezheng t biaoqian
% %% 得到优化参数
% gam = x(1);
% sig = x(2);
% M = length(train_EMG_labels);
% %% 参数设置
% type = 'c';                  % 模型类型分类
% kernel= 'RBF_kernel';        % RBF 核函数
% preprocess = 'original';     % 是否归一化
% %% 编码多分类标签
% Labels = train_EMG_labels;
% [train_EMG_labels_encoded,codebook,old_codebook] = code(train_EMG_labels,'code_OneVsOne');
% %% 建立模型
% model=initlssvm( train_EMG,train_EMG_labels_encoded,type,gam,sig,kernel,preprocess);
% model=trainlssvm(model);
% predict_label=simlssvm(model, train_EMG);
% predict_label = code(predict_label,old_codebook,[],codebook);%解码分类结果
% fitness = 1-sum((predict_label==Labels))/M;
%模型训练
%model = trainlssvm(model);
% 预测
%t_sim = simlssvm(model, train_EMG);
%得到适应度值
%fitness = sqrt(mse(t_sim - t_train));
%fitness = sum(t_sim ~= train_EMG_labels)/length(train_EMG_labels)*100;
%% Get optimization parameters
    gam = x(1);
    sig = x(2);

    type = 'c';                  % Model type: classification
    kernel = 'RBF_kernel';        % RBF kernel function
    preprocess = 'original';   % Whether to normalize the data
    %% Set the number of folds for cross-validation
    L = 10;

    %% Initialize accuracy array
    accuracies = zeros(1, L);
    indices = crossvalind('Kfold', train_EMG_labels, L);
    for i = 1:L
        %% Perform manual cross-validation
        test_indices = (indices == i);
        train_indices = ~test_indices;
        %% Get training data and test data
        train_data=train_EMG(train_indices, :);
        train_label=train_EMG_labels(train_indices);
        test_data=train_EMG(test_indices, :);
        test_label=train_EMG_labels(test_indices);
        %% Build the model
        % Encode multi-class labels
        [train_labels_encoded,codebook,old_codebook] = code(train_label, 'code_OneVsOne');
        model = initlssvm(train_data, train_labels_encoded, type, gam, sig, kernel, preprocess);

        %% Train the model
        model = trainlssvm(model);

        %% Predict
        t_sim = simlssvm(model, test_data);

        %% Decode the prediction results
        predict_label = code(t_sim,old_codebook,[],codebook);%解码分类结果

        %% Calculate accuracy for this fold
        accuracy = sum(predict_label == test_label)/ length( test_label) * 100;
        accuracies(i) = accuracy;
    end
    fitness = mean(accuracies);
    fitness=-fitness;
end

