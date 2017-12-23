

[rawX, label] = readdata();
N = size(rawX,1);
k=5:5:15;
times = size(k,2);
accur = zeros(1,times);
for i=1:times
    fprintf('\n---------------\n');
    fprintf('Running Kmeans original dataset with %d clusters\n',k(i));
    predict = runclustering(rawX, label, k(i));
    
    accur(1,i) = sum(predict == label)/N;
end

figure;
plot(k, accur);
title('K-Means');
xlabel('No of Cluster Centers');
ylabel('Classification Accuracy');