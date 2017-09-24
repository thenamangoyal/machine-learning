function task9( )
    
    data=task1();
    frac = linspace(0.1,0.9,5);    
    lambda = linspace(0,1,20);
    times = 100;
    mean_train_error = zeros(size(frac,2),size(lambda,2));
    mean_test_error = zeros(size(frac,2),size(lambda,2));
    for i = 1:size(frac,2)
        train_size = fix(frac(i)*size(data.X,1));
        for t=1:times
            rand_index=randperm(size(data.X,1));
            
            train_set.X = data.X(sort(rand_index(1:train_size)),:);
            train_set.Y = data.Y(sort(rand_index(1:train_size)),:);
            test_set.X = data.X(sort(rand_index(train_size + 1:end)),:);
            test_set.Y = data.Y(sort(rand_index(train_size + 1:end)),:);
            [train_set, mean_data, standard_deviation_data] = task2(train_set);
            test_set.X = (test_set.X - mean_data)./standard_deviation_data;
            test_set.X = [ones(size(test_set.X,1),1) test_set.X];
            for j=1:size(lambda,2)
                learn_weight = mylinridgereg(train_set.X, train_set.Y, lambda(j));
                train_predict = mylinridgeregeval(train_set.X, learn_weight);
                test_predict = mylinridgeregeval(test_set.X, learn_weight);
                mean_train_error(i,j) = mean_train_error(i,j) + meansquarederr(train_predict, train_set.Y);
                mean_test_error(i,j) = mean_test_error(i,j) + meansquarederr(test_predict, test_set.Y);
            end
        end
    end
    mean_train_error = mean_train_error/times;
    mean_test_error = mean_test_error/times;
    [min_error, lambda_pos] = min(mean_test_error');
   
    subplot(1,2,1),plot(frac, min_error), title('Min Test Error'), xlabel('Fraction'), ylabel('Min MSE');
    subplot(1,2,2),plot(frac, lambda(lambda_pos)), title('Lamda - Min Test Error'), xlabel('Fraction'), ylabel('Lambda - Min MSE');
end
