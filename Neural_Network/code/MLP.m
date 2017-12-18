function [trainerror, valerror,predict] = MLP(trainX,trainY,valX,valY,testX,nEpochs, eta, m, p)
rng('default');

% nEpochs = 1000;
% eta = 0.01; % learning rate
% p = 1; % dropout - keep with p
% m = 64; %minibatch size


fprintf('eta: %f dropout p: %f minibatch: %d\n', eta, p,m);

N = size(trainX,1); % number of training data points
no_param = size(trainX,2); % number of inputs % excluding the bias term
K = size(trainY,2); % number of outputs
H = [no_param 512 64 K]; % number of hidden layer units

% weights for the connections between input and hidden layer
% lay(1).w is a Hx(D+1) matrix
nl = size(H,2); % of layers including input
clear lay;
lay(nl-1) = struct();
for i=1:nl-1
%     lay(i).w = gpuArray(eps+(sqrt(H(i)+1))*randn(H(i+1),(H(i)+1)));
%     lay(i).w = gpuArray(-0.3+(0.6)*rand(H(i+1),(H(i)+1)));
    lay(i).w = gpuArray((-0.01)+(0.02)*rand(H(i+1),(H(i)+1)));
    lay(i).w(:,1) = 0;
end
del = (cell(nl,1));
D = (cell(nl-1,1));
z = (cell(nl,1));
% randomize the order in which the input data points are presented to the
% MLP
iporder = randperm(N);

% mlp training through stochastic gradient descent
trainerror = gpuArray(zeros(nEpochs,1));
valerror = gpuArray(zeros(nEpochs,1));
for epoch = 1:nEpochs
    tep = tic;
    times = ceil(N/m);
    for b=1:times
        no_b = m;
        if b == times
            border = (iporder(m*(b-1)+1:N));
            no_b = N - m*(b-1);
        else
            border = (iporder(m*(b-1)+1:m*b));
        end
        for i=1:nl-1
            D{i} = 0;
        end
    
        % forward pass     

        z{1} = trainX(border,:)';
        mask = gpuArray(rand(size(z{1})));
        mask = (mask <= p);
        z{1} = z{1}.*mask;

        % calculate the output of the hidden layer units - z
        for i=1:nl-2
            z{i+1} = (sigmf(lay(i).w*[ones(1,no_b);z{i}],[1,0])); % (H+1,1) matrix
            mask = gpuArray(rand(size(z{i+1})));
            mask = (mask <= p);
            z{i+1} = z{i+1}.*mask;
        end
        % hidden to output layer
        z{nl} = (lay(nl-1).w*[ones(1,no_b);z{nl-1}]);
    
        
        % backward pass
        yn = (trainY(border,:)'); % yn is (K,1) matrix - ouput
        % error in output
        del{nl} = z{nl} - yn;  % (K,1) matrix
        

        for i=nl-1:-1:2
            del{i} = (((lay(i).w(:,2:end))')*del{i+1}.*(z{i}).*(1-z{i})); %(H,1) matrix
        end


        for i=nl-1:-1:1
            D{i} = (D{i} + del{i+1}*([ones(1,no_b);z{i}]'));
            
        end
        % update the weights for the connections 
        for i=nl-1:-1:1    
            lay(i).w = (lay(i).w - (eta)*(D{i}));
        end
    end

    % compute the training error
    % forward pass of the network
    
    trainZ = (cell(nl,1));
    trainZ{1} = (p*trainX');
    for i=1:nl-2
        trainZ{i+1} = (p*sigmf(lay(i).w*([ones(1,N);trainZ{i}]),[1,0]));
    end
    trainZ{nl} = (lay(nl-1).w*[ones(1,N);trainZ{nl-1}]);
    trainZ{nl} = trainZ{nl}';
    
    valZ = (cell(nl,1));
    valN = size(valX,1);
    valZ{1} = (p*valX');
    for i=1:nl-2
        valZ{i+1} = (p*sigmf(lay(i).w*([ones(1,valN);valZ{i}]),[1,0]));
    end
    valZ{nl} = (lay(nl-1).w*[ones(1,valN);valZ{nl-1}]);
    valZ{nl} = valZ{nl}';

    trainerror(epoch) = (1/2)*(norm(trainY-trainZ{nl}).^2);
    valerror(epoch) = (1/2)*(norm(valY-valZ{nl}).^2);
    if isnan(trainerror(epoch))
        trainerror(epoch) = Inf;
    end
    if isnan(valerror(epoch))
        valerror(epoch) = Inf;
    end
   fprintf('epoch %d: training error (SSE): %f validation error (SSE): %f \n',epoch,trainerror(epoch),valerror(epoch));
    fprintf(' time: %f training error (MSE): %f validation error (MSE): %f',toc(tep),trainerror(epoch)/N,valerror(epoch)/valN);
   
    if any(trainerror(epoch) > trainerror(1:epoch-1))
        fprintf(' Train++');
    end
    if any(valerror(epoch) > valerror(1:epoch-1))
        fprintf(' Val++');
    end
    fprintf('\n');
end
    testZ = (cell(nl,1));
    testN = size(testX,1);
    testZ{1} = (p*testX');
    for i=1:nl-2
        testZ{i+1} = (p*sigmf(lay(i).w*([ones(1,testN);testZ{i}]),[1,0]));
    end
    testZ{nl} = (lay(nl-1).w*[ones(1,testN);testZ{nl-1}]);
    predict = testZ{nl}';
end