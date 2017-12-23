
[rawX, label] = readdata();
[~, U, transX] = runpca(rawX, 0.1);

N = size(transX,1);
k=5:5:15;
times = size(k,2);
accur = zeros(1,times);
projaccur = zeros(1,times);
for i=1:times
    fprintf('\n---------------\n');
    fprintf('Running Kmeans original dataset with %d clusters\n',k(i));
    
    predict = runclustering(rawX, label, k(i));
    
    fprintf('\n---------------\n');
    fprintf('Running Kmeans on reconstructed dataset with %d clusters\n',k(i));
    projpredict = runclustering(transX, label, k(i));
    
    accur(1,i) = sum(predict == label)/N;
    projaccur(1,i) = sum(projpredict == label)/N;
end

figure;
plot(k, accur, 'b'); hold on;
plot(k, projaccur, 'r');
legend('Original', 'Projected using PCA', 'Location', 'NorthOutside', ...
    'Orientation', 'horizontal');
title('K-Means');
xlabel('No of Cluster Centers');
ylabel('Classification Accuracy');