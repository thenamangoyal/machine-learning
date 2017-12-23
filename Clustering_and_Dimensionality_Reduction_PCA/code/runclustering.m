function [ predict] = runclustering( rawX, label ,k)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%     rng('default');
    N = size(rawX,1); % no of data points
    
    
    predict = zeros(N,1);
    
    [idx, ~] = kmeans(rawX, k);
    for n=1:k
        points_in_cluster = (idx == n);
        predict(points_in_cluster) = mode(label(points_in_cluster));
        
    end
    % Accuracy
accuracy = sum(predict == label)/N;

index_to_class = unique(label); % Map Index -> Class
no_class = size(index_to_class,1);
class_to_index = containers.Map(index_to_class, 1:no_class); % Map Class -> Index

conf_matrix = zeros(no_class, no_class);
for i=1:N
    conf_matrix(class_to_index(label(i)),class_to_index(predict(i))) = conf_matrix(class_to_index(label(i)),class_to_index(predict(i))) +1;
end

classwise_precision = zeros(no_class,1);
classwise_recall = zeros(no_class,1);
    
for i=1:no_class
    
    if sum(conf_matrix(:,i)) ~= 0
        classwise_precision(i,1) = conf_matrix(i,i)/sum(conf_matrix(:,i)); % Positive/ (No of Predictied positive)
    else
        classwise_precision(i,1) = 0;
    end
    if sum(conf_matrix(i,:)) ~= 0
        classwise_recall(i,1) = conf_matrix(i,i)/sum(conf_matrix(i,:));  % Positive/ (No of Actual positive)
    else
        classwise_recall(i,1) = 0;
    end
    
end    

fprintf('Confusion Matrix (Row: Actual,Col: Prediction)\n');
disptable(conf_matrix,cellstr(num2str(index_to_class)),cellstr(num2str(index_to_class)));
disp(table(index_to_class,classwise_precision,classwise_recall,'VariableNames',{'Class','Precision', 'Recall'}))

% (csvwrite('temp.csv',[0 index_to_class';index_to_class conf_matrix ]))
% type temp.csv
% fprintf('\nClass,Precision,Recall');
% (csvwrite('temp.csv',[index_to_class, classwise_precision,classwise_recall]))
% type temp.csv

fprintf('Overall Accuracy %f\n', accuracy);
end

