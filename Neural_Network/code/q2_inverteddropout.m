close all;
clear all;
% rng('default');
% Getdata
re_read = treu;
if re_read==true
   [X,Y] = gettraindata(fullfile('.','steering'),'data.txt');
   save('wholeTraindata.mat','X','Y'); % no separation in validation or train set
else
   load('wholeTraindata.mat','X','Y')
end

% Create Train and Validation
% rng('default');
sep = tic; 	
N = size(X,1);
split = randperm(N);
split_n = floor((0.8)*N);
trainX = gpuArray(X(split(1:split_n),:));
trainY = gpuArray(Y(split(1:split_n),:));
valX = gpuArray(X(split(split_n+1:N),:));
valY = gpuArray(Y(split(split_n+1:N),:));
fprintf('Create train and validation set in %f sec\n',toc(sep));

%% Read Test Data
re_read_test = true;
if re_read_test==true
   [testX] = gettestdata(fullfile('.'),'test-data.txt');
   save('wholeTestdata.mat','testX'); % no separation in validation or train set
else
   load('wholeTestdata.mat','testX')
end

%% train the MLP using the generated sample dataset
% X - training data of size NxD
% Y - training labels of size NxK
% H - the number of hiffe

nEpochs = 1;
eta = 0.05; % learning rate


m = 64; %minibatch size

fprintf('eta: %f dropout minibatch: %d\n', eta,m);

N = size(trainX,1); % number of training data points
no_param = size(trainX,2); % number of inputs % excluding the bias term
K = size(trainY,2); % number of outputs
H = [no_param 512 64 K]; % number of hidden layer units
% orgp = [1 0.5 0.8 1]; % dropout - keep with p


% weights for the connections between input and hidden layer
% lay(1).w is a Hx(D+1) matrix
nl = size(H,2); % of layers including input

clear lay;
Sdw = cell(nl-1,1);
Sdb = cell(nl-1,1);
b2 =  0.999;

Vdw = cell(nl-1,1);
Vdb = cell(nl-1,1);
b1 = 0.9;

lay(nl-1) = struct();
for i=1:nl-1
    
    temp = ((2/(sqrt(H(i)+H(i+1))))*randn(H(i+1),(H(i)+1)));
    lay(i).b = temp(:,1);
    lay(i).w = temp(:,2:end);
    Sdw{i} = zeros(size(lay(i).w));
    Sdb{i} = zeros(size(lay(i).b));
    Vdw{i} = zeros(size(lay(i).w));
    Vdb{i} = zeros(size(lay(i).b));
end



del = (cell(nl,1));
Dw = (cell(nl-1,1));
Db = (cell(nl-1,1));
z = (cell(nl,1));

% randomize the order in which the input data points are presented to the
% MLP

iporder = randperm(N);
trainerror = gpuArray(zeros(nEpochs,1));
valerror = gpuArray(zeros(nEpochs,1));
for epoch = 1:nEpochs    
    tep = tic;
    times = ceil(N/m);
    newp = [1 sigmf(epoch,[1,0]) 1];
    for b=1:times
        no_b = m;
        if b == times
            border = (iporder(m*(b-1)+1:N));
            no_b = N - m*(b-1);
        else
            border = (iporder(m*(b-1)+1:m*b));
        end
        for i=1:nl-1
            Dw{i} = 0;
            Db{i} = 0;
        end
    
        % forward pass
        z{1} = trainX(border,:)';
        mask = gpuArray(rand(size(z{1})));
        mask = (mask <= newp(1));
        z{1} = (1/newp(1))*z{1}.*mask;

        % calculate the output of the hidden layer units - z
        for i=1:nl-2
            z{i+1} = (sigmf(repmat(lay(i).b,1,no_b)+lay(i).w*[z{i}],[1,0])); % (H+1,1) matrix
            mask = gpuArray(rand(size(z{i+1})));
            mask = (mask <= newp(i));
            z{i+1} = (1/newp(i+1))*z{i+1}.*mask;
        end
        % hidden to output layer
        z{nl} = repmat(lay(nl-1).b,1,no_b)+(lay(nl-1).w*[z{nl-1}]);
        
        % backward pass
        yn = (trainY(border,:)'); % yn is (K,1) matrix - ouput
        % error in output
        del{nl} = z{nl} - yn;  % (K,1) matrix

        for i=nl-1:-1:2
            del{i} = ((lay(i).w')*del{i+1}.*(z{i}).*(1-z{i})); %(H,1) matrix
        end


        for i=nl-1:-1:1
            Db{i} = Db{i} + sum(del{i+1},2); % Sum along dimension of batch size
            Dw{i} = (Dw{i} + del{i+1}*(z{i}'));

            
        end
        % update the weights for the connections 
        for i=nl-1:-1:1
            Sdw{i} = b2*Sdw{i} + (1-b2)*(Dw{i}.*Dw{i});
            Sdb{i} = b2*Sdb{i} + (1-b2)*(Db{i}.*Db{i});
            Vdw{i} = b1*Vdw{i} + (1-b1)*(Dw{i});
            Vdb{i} = b1*Vdb{i} + (1-b1)*(Db{i});
            Vdwcorrected = Vdw{i}/(1-b1^epoch);
            Vdbcorrected = Vdb{i}/(1-b1^epoch);
            Sdwcorrected = Sdw{i}/(1-b2^epoch);
            Sdbcorrected = Sdb{i}/(1-b2^epoch);
            lay(i).w = lay(i).w - (eta/no_b)*(Vdwcorrected./(sqrt(Sdwcorrected)+eps));
            lay(i).b = lay(i).b - (eta/no_b)*(Vdbcorrected./(sqrt(Sdbcorrected)+eps));
        end
    end

    % compute the training error
    % forward pass of the network
    
    trainout = testgpu(trainX,lay,nl);
    
    valout = testgpu(valX,lay,nl);
    
    trainerror(epoch) = (norm(trainY-trainout).^2);
    valerror(epoch) = (norm(valY-valout).^2);
    fprintf('epoch %d: training error: %f validation error: %f \n',epoch,trainerror(epoch),valerror(epoch));
    fprintf('\t time: %f training MSE: %f validation MSE: %f',toc(tep),trainerror(epoch)/size(trainX,1),valerror(epoch)/size(valX,1));
   
    if any(trainerror(epoch) > trainerror(1:epoch-1))
        fprintf(' Train++ minTrain %f',min(trainerror(1:epoch-1)));
    end
    if (any(valerror(epoch) > valerror(1:epoch-1)))
        fprintf(' Val++ minVal %f',min(valerror(1:epoch-1)));
        
    else
        testbestprediction = testgpu(testX,lay,nl);
        dlmwrite((['predict/',num2str(valerror(epoch)),'_epoch_',num2str(epoch),'_','.txt']),testbestprediction,'precision',12);
    end
    fprintf('\n');
end
% Predict Z

dlmwrite('prediction-test3.txt',testbestprediction,'precision',12);


