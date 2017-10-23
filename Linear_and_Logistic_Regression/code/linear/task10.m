function task10( )
    
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
    [min_error_by_lambda, frac_pos] = min(mean_test_error);
    [min_error, lambda_pos] = min(min_error_by_lambda);
    min_frac_pos = frac_pos(lambda_pos);
    min_lambda_pos = lambda_pos;
    
    train_size = fix(frac(min_frac_pos)*size(data.X,1));
    rand_index=randperm(size(data.X,1));            
    train_set.X = data.X(sort(rand_index(1:train_size)),:);
    train_set.Y = data.Y(sort(rand_index(1:train_size)),:);
    test_set.X = data.X(sort(rand_index(train_size + 1:end)),:);
    test_set.Y = data.Y(sort(rand_index(train_size + 1:end)),:);
    [train_set, mean_data, standard_deviation_data] = task2(train_set);
    test_set.X = (test_set.X - mean_data)./standard_deviation_data;
    test_set.X = [ones(size(test_set.X,1),1) test_set.X];
    learn_weight = mylinridgereg(train_set.X, train_set.Y, lambda(min_lambda_pos));
    train_predict = mylinridgeregeval(train_set.X, learn_weight);
    
    test_predict = mylinridgeregeval(test_set.X, learn_weight);
    limits = [floor(min([min(min(train_predict)) min(min(test_predict)) min(min(train_set.Y))])) ceil(max([max(max(train_predict)) max(max(test_predict)) max(max(train_set.Y))]))];
    eq_line=limits(1):1:limits(2);
    train_fit = polyfit(train_predict, train_set.Y,1);
    train_fit_y = polyval(train_fit, eq_line);
    test_fit = polyfit(test_predict, test_set.Y,1);
    test_fit_y = polyval(test_fit, eq_line);
    subplot(1,2,1),plot(train_predict, train_set.Y,'b.',eq_line,eq_line,'g',eq_line, train_fit_y,'r'), title('Train Plot'), xlabel('Predicted'), ylabel('Actual'), xlim(limits), ylim(limits),
    legend('Data Point','Actual Y=X','Trend from model','Location','northwest');
    legend('boxoff');
    subplot(1,2,2),plot(test_predict, test_set.Y,'b.',eq_line,eq_line,'g',eq_line, test_fit_y,'r'), title('Test Plot'), xlabel('Predicted'), ylabel('Actual'), xlim(limits), ylim(limits),
    legend('Data Point','Actual Y=X','Trend from model','Location','northwest');
    legend('boxoff');
end
