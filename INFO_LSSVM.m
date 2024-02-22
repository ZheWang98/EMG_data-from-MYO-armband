%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行
%% 
filename = 'Feature_WZ_128D'; % 
sheet = 1; % 读取的工作表索引或名称
data = readtable(filename, 'Sheet', sheet);

% 根据 CLASS 列的不同值，提取前 80% 和剩余 20% 数据
uniqueClasses = unique(data.class); % 获取唯一的 CLASS 值
percentage = 0.9;

% 初始化一个用于存储提取数据的表格
extractedData = table();
remainingData = table();
%循环读取数据
for i = 1:length(uniqueClasses)
    className = uniqueClasses(i);
    [classExtractedData, classRemainingData] = extractDataByClass(data, className, percentage);
    extractedData = [extractedData; classExtractedData];
    remainingData = [remainingData; classRemainingData];
end
train_EMG = extractedData{:, 1:128};
train_EMG_labels = extractedData{:, 129};
test_EMG = remainingData{:, 1:128};
test_EMG_labels = remainingData{:, 129};
%%  参数设置
pop = 20;              % 种群数目
Max_iter = 100;         % 迭代次数
dim = 2;               % 优化参数个数
lb = [0.1,0.1];       % 下限
ub = [1000,10];       % 上限
type = 'c';
kernel_type = 'RBF_kernel';
codefct = 'code_OneVsOne'; 
preprocess = 'original';%'preprocess'or'original'
%% 优化函数
fobj = @(x)fitnessfunclssvm(x, train_EMG, train_EMG_labels);
%% 优化
[Best_cost, Best_pos, curve,avcurve] = INFO(pop, Max_iter, lb, ub, dim, fobj);
f1=figure(1);
hold on;
plot(-curve,'r-','LineWidth',1.5);
plot(-avcurve,'b-','LineWidth',1.5);
%legend('Best fitness','Average fitness','FontSize',12,'FontName','Times New Roman','Location','southeast');
legend('最佳适应度','平均适应度','FontSize',12,'FontName','宋体','Location','southeast');
%xlabel('The number of iteration','FontSize',13,'FontName','Times New Roman');
xlabel('迭代次数','FontSize',13,'FontName','宋体');
%ylabel('Fitness value','FontSize',13,'FontName','Times New Roman');
ylabel('适应度','FontSize',13,'FontName','宋体');
set(gca,'XMinorTick',true)
set(gca,'YMinorTick',true)
grid on; 
box on;
hold off
exportgraphics(f1,'fitness_74.4889_ALL.emf', 'Resolution',600)

%% 建立模型
[train_EMG_labels_encode, codebook, old_codebook] = code(train_EMG_labels, codefct);%编码
gam = Best_pos(1);  
sig = Best_pos(2);
model = initlssvm(train_EMG,train_EMG_labels_encode,type,gam,sig,kernel_type,preprocess); 
model = trainlssvm(model);%训练模型
%% 模型预测
% 训练集准确率
predict_label_train=simlssvm(model,train_EMG);
predict_label_train = code(predict_label_train,old_codebook,[],codebook);%解码分类结果
total = length(train_EMG_labels);
right = sum(predict_label_train == train_EMG_labels);
accuracy_train = right / total * 100;
fprintf('Training accuracy: %.2f%%\n', accuracy_train);
% 测试集准确率
predict_label_test = simlssvm(model,test_EMG);
predict_label_test = code(predict_label_test,old_codebook,[],codebook);%解码分类结果
total = length(test_EMG_labels);
right = sum(predict_label_test == test_EMG_labels);
accuracy_test = right / total * 100;
fprintf('Testing accuracy: %.2f%%\n', accuracy_test);

confusionMatrix_test = confusionmat(test_EMG_labels, predict_label_test);
% 显示混淆矩阵
disp('Confusion Matrix:');
disp(confusionMatrix_test);
f1=figure(1);
%confusionchart(confusionMatrix_test, {'Fist','Pinch','TWO','Three','Good','Tripod','Spherical','HO','INDEX','Cylindrical'},'Normalization', 'row-normalized'); % 根据你的类别修改标签
confusionchart(confusionMatrix_test, {'握拳手势','二指捏取','手势2','手势3','点赞','三指捏取','球形手势','展开手势','指向手势','柱形手势'},'Normalization', 'row-normalized'); % 根据你的类别修改标签
%title('Confusion Matrix');
%title('混淆矩阵','FontSize',12,'FontName','宋体');
%xlabel('Predicted Class');
%xlabel('预测类别','FontSize',12,'FontName','宋体');
%ylabel('True Class')
%ylabel('真实类别','FontSize',12,'FontName','宋体');
%set(gca,'FontName','Times New Roman','FontSize',12)
exportgraphics(f1,'confusionMatrix_83.6%_AR+TD+RMS.emf', 'Resolution',600)

% %%  数据反归一化
% T_sim1 = mapminmax('reverse', t_sim1, ps_output);
% T_sim2 = mapminmax('reverse', t_sim2, ps_output);
% 
% %%  均方根误差
% error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
% error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);
% 
% %% 优化曲线
% figure
% plot(curve, 'linewidth', 1.5);
% title('INFO-LSSVM Iterative curve')
% xlabel('The number of iterations')
% ylabel('Fitness')
% grid on;
% 
%%  绘图
% figure
% plot(1: M, train_EMG_labels, 'r-*', 1: M, predict_label_train, 'b-o', 'LineWidth', 1)
% legend('真实值','预测值')
% xlabel('预测样本')
% ylabel('预测结果')
% string = {'训练集预测准确率'; ['Accuracy=' num2str(accuracy_train)]};
% title(string)
% xlim([1, M])
% grid
% 
% figure
% plot(1: N, test_EMG_labels, 'r-*', 1: N, predict_label_test, 'b-o', 'LineWidth', 1)
% legend('真实值','预测值')
% xlabel('预测样本')
% ylabel('预测结果')
% string = {'测试集预测准确率';['Accuracy=' num2str(accuracy_test)]};
% title(string)
% xlim([1, N])
% grid
% 
% %%  相关指标计算
% %  R2
% R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
% R2 = 1 - norm(T_test -  T_sim2')^2 / norm(T_test -  mean(T_test ))^2;
% 
% disp(['训练集数据的R2为：', num2str(R1)])
% disp(['测试集数据的R2为：', num2str(R2)])
% 
% %  MAE
% mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
% mae2 = sum(abs(T_sim2' - T_test )) ./ N ;
% 
% disp(['训练集数据的MAE为：', num2str(mae1)])
% disp(['测试集数据的MAE为：', num2str(mae2)])
% 
% %  MBE
% mbe1 = sum(T_sim1' - T_train) ./ M ;
% mbe2 = sum(T_sim2' - T_test ) ./ N ;
% 
% disp(['训练集数据的MBE为：', num2str(mbe1)])
% disp(['测试集数据的MBE为：', num2str(mbe2)])