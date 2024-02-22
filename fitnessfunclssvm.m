function fitness = fitnessfunclssvm(x, train_EMG, train_EMG_labels)
% %% ������Ӧ�Ⱥ���
% %p tezheng t biaoqian
% %% �õ��Ż�����
% gam = x(1);
% sig = x(2);
% M = length(train_EMG_labels);
% %% ��������
% type = 'c';                  % ģ�����ͷ���
% kernel= 'RBF_kernel';        % RBF �˺���
% preprocess = 'original';     % �Ƿ��һ��
% %% ���������ǩ
% Labels = train_EMG_labels;
% [train_EMG_labels_encoded,codebook,old_codebook] = code(train_EMG_labels,'code_OneVsOne');
% %% ����ģ��
% model=initlssvm( train_EMG,train_EMG_labels_encoded,type,gam,sig,kernel,preprocess);
% model=trainlssvm(model);
% predict_label=simlssvm(model, train_EMG);
% predict_label = code(predict_label,old_codebook,[],codebook);%���������
% fitness = 1-sum((predict_label==Labels))/M;
%ģ��ѵ��
%model = trainlssvm(model);
% Ԥ��
%t_sim = simlssvm(model, train_EMG);
%�õ���Ӧ��ֵ
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
        predict_label = code(t_sim,old_codebook,[],codebook);%���������

        %% Calculate accuracy for this fold
        accuracy = sum(predict_label == test_label)/ length( test_label) * 100;
        accuracies(i) = accuracy;
    end
    fitness = mean(accuracies);
    fitness=-fitness;
end

