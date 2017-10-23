function [ output_args ] = task2( )
    close all;
    load('credit.mat');
    lambda = 0;
    alpha = 1;
    limit_iter = 1000000;
    X = [ones(size(data,1),1) data];
%% Gradient Descent
    fprintf('Gradient Descent\n');
    W = zeros(size(X,2),1);
    torun = 1;
    count = 0;
    while (count < limit_iter) && (torun == 1) && (sum((X*W>0)==label)/size(X,1) < 1-eps)
        non_W = W;
        if size(non_W,1) >= 1 && size(non_W,2) >= 1
            non_W(1,1) = 0;
        end
        Wnew = W - alpha*(1/(size(X,1)))*((X')*(sigmf(X*W,[1 0]) - label) + lambda*non_W);
        if max(abs(Wnew -W)) > 0.05
            torun = 1;
        else
            torun = 0;
        end
        W = Wnew;
        count = count+1;
    end
    fprintf("No of Iterations :%d\n",count);
    fprintf("Accuracy :%f\n",sum((X*W>0)==label)/size(X,1));
    
    plot_x = linspace(min(X(:,2)),max(X(:,2)),100);
    plot_y = linspace(min(X(:,3)),max(X(:,3)),100);
    plot_z = zeros(size(plot_y,2),size(plot_x,2));
    for i=1:size(plot_y,2)
        for j=1:size(plot_x,2)
            plot_z(i,j) = [1 plot_x(1,j) plot_y(1,i)]*W;
        end
    end    
    figure;
    subplot(1,2,1);
    plot(X(find(label==1),2),X(find(label==1),3),'b+',X(find(label==0),2),X(find(label==0),3),'ro'),
    hold on,
    contour(plot_x,plot_y,plot_z,[0 0],'g','LineWidth',2),
    title('Gradient Descent'),
    legend('Positive','Negative','Decision Boundary');
%% Newton Raphson
    fprintf('\nNewton Raphson\n');
    W = zeros(size(X,2),1);
    torun = 1;
    count = 0;
    
    limit_iter = 1000000/100;
    while (count < limit_iter) && (torun == 1) && (sum((X*W>0)==label)/size(X,1) < 1-eps)
        fx = sigmf(X*W,[1 0]);
        R = (diag(fx))*(diag(1-fx));
        Z = X*W - pinv(R)*(fx-label);
        I = eye(size(X,2));
        if size(I,1) >= 1 && size(I,2) >= 1
            I(1,1) = 0;
        end
        Wnew = pinv((X')*R*X + lambda*I)*(X')*R*Z;
        if max(abs(Wnew -W)) > 0.05
            torun = 1;
        else
            torun = 0;
        end
        W = Wnew;
        count = count+1;
    end
    
    fprintf("No of Iterations :%d\n",count);
    fprintf("Accuracy :%f\n",sum((X*W>0)==label)/size(X,1));
    
    plot_x = linspace(min(X(:,2)),max(X(:,2)),100);
    plot_y = linspace(min(X(:,3)),max(X(:,3)),100);
    plot_z = zeros(size(plot_y,2),size(plot_x,2));
    for i=1:size(plot_y,2)
        for j=1:size(plot_x,2)
            plot_z(i,j) = [1 plot_x(1,j) plot_y(1,i)]*W;
        end
    end
    subplot(1,2,2);
    plot(X(find(label==1),2),X(find(label==1),3),'b+',X(find(label==0),2),X(find(label==0),3),'ro'),
    hold on,
    contour(plot_x,plot_y,plot_z,[0 0],'g','LineWidth',2),
    title('Newton Raphson'),
    legend('Positive','Negative','Decision Boundary');
end