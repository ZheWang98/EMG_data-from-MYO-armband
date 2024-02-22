%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������
%% 
filename = 'Feature_WZ_128D'; % 
sheet = 1; % ��ȡ�Ĺ���������������
data = readtable(filename, 'Sheet', sheet);

% ���� CLASS �еĲ�ֵͬ����ȡǰ 80% ��ʣ�� 20% ����
uniqueClasses = unique(data.class); % ��ȡΨһ�� CLASS ֵ
percentage = 0.9;

% ��ʼ��һ�����ڴ洢��ȡ���ݵı��
extractedData = table();
remainingData = table();
%ѭ����ȡ����
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
%%  ��������
pop = 20;              % ��Ⱥ��Ŀ
Max_iter = 100;         % ��������
dim = 2;               % �Ż���������
lb = [0.1,0.1];       % ����
ub = [1000,10];       % ����
type = 'c';
kernel_type = 'RBF_kernel';
codefct = 'code_OneVsOne'; 
preprocess = 'original';%'preprocess'or'original'
%% �Ż�����
fobj = @(x)fitnessfunclssvm(x, train_EMG, train_EMG_labels);
%% �Ż�
[Best_cost, Best_pos, curve,avcurve] = INFO(pop, Max_iter, lb, ub, dim, fobj);
f1=figure(1);
hold on;
plot(-curve,'r-','LineWidth',1.5);
plot(-avcurve,'b-','LineWidth',1.5);
%legend('Best fitness','Average fitness','FontSize',12,'FontName','Times New Roman','Location','southeast');
legend('�����Ӧ��','ƽ����Ӧ��','FontSize',12,'FontName','����','Location','southeast');
%xlabel('The number of iteration','FontSize',13,'FontName','Times New Roman');
xlabel('��������','FontSize',13,'FontName','����');
%ylabel('Fitness value','FontSize',13,'FontName','Times New Roman');
ylabel('��Ӧ��','FontSize',13,'FontName','����');
set(gca,'XMinorTick',true)
set(gca,'YMinorTick',true)
grid on; 
box on;
hold off
exportgraphics(f1,'fitness_74.4889_ALL.emf', 'Resolution',600)

%% ����ģ��
[train_EMG_labels_encode, codebook, old_codebook] = code(train_EMG_labels, codefct);%����
gam = Best_pos(1);  
sig = Best_pos(2);
model = initlssvm(train_EMG,train_EMG_labels_encode,type,gam,sig,kernel_type,preprocess); 
model = trainlssvm(model);%ѵ��ģ��
%% ģ��Ԥ��
% ѵ����׼ȷ��
predict_label_train=simlssvm(model,train_EMG);
predict_label_train = code(predict_label_train,old_codebook,[],codebook);%���������
total = length(train_EMG_labels);
right = sum(predict_label_train == train_EMG_labels);
accuracy_train = right / total * 100;
fprintf('Training accuracy: %.2f%%\n', accuracy_train);
% ���Լ�׼ȷ��
predict_label_test = simlssvm(model,test_EMG);
predict_label_test = code(predict_label_test,old_codebook,[],codebook);%���������
total = length(test_EMG_labels);
right = sum(predict_label_test == test_EMG_labels);
accuracy_test = right / total * 100;
fprintf('Testing accuracy: %.2f%%\n', accuracy_test);

confusionMatrix_test = confusionmat(test_EMG_labels, predict_label_test);
% ��ʾ��������
disp('Confusion Matrix:');
disp(confusionMatrix_test);
f1=figure(1);
%confusionchart(confusionMatrix_test, {'Fist','Pinch','TWO','Three','Good','Tripod','Spherical','HO','INDEX','Cylindrical'},'Normalization', 'row-normalized'); % �����������޸ı�ǩ
confusionchart(confusionMatrix_test, {'��ȭ����','��ָ��ȡ','����2','����3','����','��ָ��ȡ','��������','չ������','ָ������','��������'},'Normalization', 'row-normalized'); % �����������޸ı�ǩ
%title('Confusion Matrix');
%title('��������','FontSize',12,'FontName','����');
%xlabel('Predicted Class');
%xlabel('Ԥ�����','FontSize',12,'FontName','����');
%ylabel('True Class')
%ylabel('��ʵ���','FontSize',12,'FontName','����');
%set(gca,'FontName','Times New Roman','FontSize',12)
exportgraphics(f1,'confusionMatrix_83.6%_AR+TD+RMS.emf', 'Resolution',600)

% %%  ���ݷ���һ��
% T_sim1 = mapminmax('reverse', t_sim1, ps_output);
% T_sim2 = mapminmax('reverse', t_sim2, ps_output);
% 
% %%  ���������
% error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
% error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);
% 
% %% �Ż�����
% figure
% plot(curve, 'linewidth', 1.5);
% title('INFO-LSSVM Iterative curve')
% xlabel('The number of iterations')
% ylabel('Fitness')
% grid on;
% 
%%  ��ͼ
% figure
% plot(1: M, train_EMG_labels, 'r-*', 1: M, predict_label_train, 'b-o', 'LineWidth', 1)
% legend('��ʵֵ','Ԥ��ֵ')
% xlabel('Ԥ������')
% ylabel('Ԥ����')
% string = {'ѵ����Ԥ��׼ȷ��'; ['Accuracy=' num2str(accuracy_train)]};
% title(string)
% xlim([1, M])
% grid
% 
% figure
% plot(1: N, test_EMG_labels, 'r-*', 1: N, predict_label_test, 'b-o', 'LineWidth', 1)
% legend('��ʵֵ','Ԥ��ֵ')
% xlabel('Ԥ������')
% ylabel('Ԥ����')
% string = {'���Լ�Ԥ��׼ȷ��';['Accuracy=' num2str(accuracy_test)]};
% title(string)
% xlim([1, N])
% grid
% 
% %%  ���ָ�����
% %  R2
% R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
% R2 = 1 - norm(T_test -  T_sim2')^2 / norm(T_test -  mean(T_test ))^2;
% 
% disp(['ѵ�������ݵ�R2Ϊ��', num2str(R1)])
% disp(['���Լ����ݵ�R2Ϊ��', num2str(R2)])
% 
% %  MAE
% mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
% mae2 = sum(abs(T_sim2' - T_test )) ./ N ;
% 
% disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1)])
% disp(['���Լ����ݵ�MAEΪ��', num2str(mae2)])
% 
% %  MBE
% mbe1 = sum(T_sim1' - T_train) ./ M ;
% mbe2 = sum(T_sim2' - T_test ) ./ N ;
% 
% disp(['ѵ�������ݵ�MBEΪ��', num2str(mbe1)])
% disp(['���Լ����ݵ�MBEΪ��', num2str(mbe2)])