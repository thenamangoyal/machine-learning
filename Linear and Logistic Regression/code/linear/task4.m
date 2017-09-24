function task4( frac )
    if (nargin == 0) || (isscalar(frac) == 0)
        frac = 0.2;
    end
    if frac > 1
        frac = 1;
    elseif frac < 0
        frac = 0;
    end
    
    data=task1();
    rand_index=randperm(size(data.X,1));
    train_size = fix(frac*size(data.X,1));
    train_set.X = data.X(sort(rand_index(1:train_size)),:);
    train_set.Y = data.Y(sort(rand_index(1:train_size)),:);
    test_set.X = data.X(sort(rand_index(train_size + 1:end)),:);
    test_set.Y = data.Y(sort(rand_index(train_size + 1:end)),:);
    [train_set, mean_data, standard_deviation_data] = task2(train_set);
    test_set.X = (test_set.X - mean_data)./standard_deviation_data;
    test_set.X = [ones(size(test_set.X,1),1) test_set.X];
    for lambda=linspace(0,1,20)
        learn_weight = mylinridgereg(train_set.X, train_set.Y, lambda);
        train_predict = mylinridgeregeval(train_set.X, learn_weight);
        test_predict = mylinridgeregeval(test_set.X, learn_weight);
        fprintf('Lambda = %d, TrainError = %f, TestError=%f\n',lambda,meansquarederr(train_predict,train_set.Y),meansquarederr(test_predict,test_set.Y));
    end
end
