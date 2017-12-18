close all;
clear all;
% % rng('default');
% % Getdata
re_read = false;
if re_read==true
   [X,Y] = gettraindata(fullfile('.','steering'),'data.txt');
   save('wholeTraindata.mat','X','Y'); % no separation in validation or train set
else
   load('wholeTraindata.mat','X','Y')
end
% 
% Create Train and Validation
% rng('default');
sep = tic; 	

X = X(1:80,:);
Y = Y(1:80,:);
N = size(X,1);
split = randperm(N);
split_n = floor((0.8)*N);
trainX = gpuArray(X(split(1:split_n),:));
trainY = gpuArray(Y(split(1:split_n),:));
valX = gpuArray(X(split(split_n+1:N),:));
valY = gpuArray(Y(split(split_n+1:N),:));
fprintf('Create train and validation set in %f sec\n',toc(sep));
% 
% %% Read Test Data
re_read_test = false;
if re_read_test==true
   [testX] = gettestdata(fullfile('.'),'test-data.txt');
   save('wholeTestdata.mat','testX'); % no separation in validation or train set
else
   load('wholeTestdata.mat','testX')
end
% 
% %% train the MLP using the generated sample dataset
% % X - training data of size NxD
% % Y - training labels of size NxK
% % H - the number of hiffe
% 
nEpochs = 5;
eta = 0.05; % learning rate

p = 1; % dropout - keep with p
m = 10; %minibatch size

fprintf('eta: %f dropout p: %f minibatch: %d\n', eta, p,m);

N = size(trainX,1); % number of training data points
no_param = size(trainX,2); % number of inputs % excluding the bias term
K = size(trainY,2); % number of outputs

dim = ceil(sqrt(no_param));
newtrainX = permute(trainX,[2,1]);
newtrainX = reshape(newtrainX,[dim,dim,1,size(trainX,1)]);

newvalX = permute(valX,[2,1]);
newvalX = reshape(newvalX,[dim,dim,1,size(valX,1)]);
%%
nl = 7;
clear lay;
lay(nl-1) = struct();
lay(1).type = 'conv';
lay(1).no_channel = size(newtrainX,3);
lay(1).no_filter = 8;
lay(1).w = -0.1 +2*(0.1)*rand(4,4,lay(1).no_channel,lay(1).no_filter); % k1xk2xchannelxno_filter
lay(1).outdim = [size(convn(newtrainX(:,:,:,1),lay(1).w(:,:,:,1),'valid')) lay(1).no_filter]; % exclude batch

lay(2).type = 'relu';
lay(2).outdim = lay(1).outdim;

lay(3).type = 'maxpool';
lay(3).stride = 4;
lay(3).outdim = floor((lay(2).outdim)./(lay(3).stride));
lay(3).index1 = [];
lay(3).index2 = [];
lay(3).next = [];


lay(4).type = 'fully';
lay(4).outdim = 10;
temp = 2/(sqrt(prod(lay(3).outdim)+1+lay(4).outdim))*randn(lay(4).outdim,(prod(lay(3).outdim)+1));
lay(4).b = temp(:,1);
lay(4).w = temp(:,2:end);

lay(5).type = 'relu';
lay(5).outdim = lay(4).outdim;


lay(6).type = 'fully';
lay(6).outdim = 1;
temp = 2/(sqrt(prod(lay(4).outdim)+1+lay(5).outdim))*randn(lay(5).outdim,(prod(lay(4).outdim)+1));
lay(6).b = temp(:,1);
lay(6).w = temp(:,2:end);



%%
Vdw = cell(nl-1,1);
Vdb = cell(nl-1,1);
b1 = 0.9;
Sdw = cell(nl-1,1);
Sdb = cell(nl-1,1);
b2 =  0.999;
for i=1:nl-1
    
    Sdw{i} = zeros(size(lay(i).w));
    Sdb{i} = zeros(size(lay(i).b));
    Vdw{i} = zeros(size(lay(i).w));
    Vdb{i} = zeros(size(lay(i).b));
end

del = (cell(nl,1));
Dw = (cell(nl-1,1));
Db = (cell(nl-1,1));
Z = (cell(nl,1));

% randomize the order in which the input data points are presented to the
% MLP

iporder = randperm(N);
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
            Dw{i} = 0;
            Db{i} = 0;
        end
    
        % forward pass
        Z{1} = gpuArray(newtrainX(:,:,:,border));

        % calculate the output of the hidden layer units - z
        for i=1:nl-2
            Z{i+1} = gpuArray(zeros([lay(i).outdim, no_b]));
            if strcmp(lay(i).type, 'conv') ==1   
%                 tic
                for f=1:lay(i).no_filter % no filter
                   Z{i+1}(:,:,f,:) = (convn(Z{i},lay(i).w(:,:,:,f),'valid'));
                end
%                 toc
            elseif strcmp(lay(i).type, 'relu') == 1
                 Z{i+1} = max(Z{i},0);
            elseif strcmp(lay(i).type,'maxpool') ==1
%                 tic
                  lay(i).index1 = gpuArray(zeros([lay(i).outdim, no_b]));
                  lay(i).index2 = gpuArray(zeros([lay(i).outdim, no_b]));
%                   for k=1:no_b
                      for c=1:lay(i).outdim(3) % channel
                          for a=1:lay(i).outdim(1)
                              for b=1:lay(i).outdim(2)
                                  [val,rowno] = max(Z{i}((a-1)*(lay(i).stride)+1:(a)*(lay(i).stride),(b-1)*(lay(i).stride)+1:(b)*(lay(i).stride),c,:));
                                  [Z{i+1}(a,b,c,:), colno]= max(val);
                                  lay(i).index1(a,b,c,:) = rowno(colno);
                                  lay(i).index2(a,b,c,:) = colno;
                              end
                          end
                      end
%                   end
%                  fprintf('Maxpool\n');
%                  toc
            elseif strcmp(lay(i).type,'fully') ==1
%                 tic
%                 if strcmp(lay(i-1).type,'fully') == 0
                    Z{i} = reshape(Z{i}, [prod(lay(i-1).outdim) no_b]);
%                 end
                Z{i+1} = ((repmat(lay(i).b,1,no_b)+lay(i).w*[Z{i}])); % (H+1,1) matrix
                
%                  fprintf('Fully \n');
%                  toc
            end
        end
        % hidden to output layer
        Z{nl} = repmat(lay(nl-1).b,1,no_b)+(lay(nl-1).w*[Z{nl-1}]);
        
        % backward pass
        yn = (trainY(border,:)'); % yn is (K,1) matrix - ouput
        % error in output
        del{nl} = Z{nl} - yn;  % (K,1) matrix
        yn
        q = Z{nl}

        for i=nl-1:-1:2 % gives error in Z{i} using Z{i+1} where Z{i} input to layer i and Z{i+1} is output of layer i
            if strcmp(lay(i).type, 'fully') == 1
               del{i} = ((lay(i).w')*del{i+1}); %(H,1) matrix
               del{i} = reshape(del{i},size(Z{i}));
            elseif strcmp(lay(i).type, 'maxpool') == 1
                del{i+1} = reshape(del{i+1},[lay(i).outdim no_b]);
                del{i} = gpuArray(zeros(size(Z{i})));
                  for k=1:no_b
                      for c=1:lay(i).outdim(3) % channel
                          for a=1:lay(i).outdim(1)
                              for b=1:lay(i).outdim(2)
                                  dim1 = (a-1)*(lay(i).stride)+lay(i).index1(a,b,c,k);
                                  dim2 = (b-1)*(lay(i).stride)+lay(i).index2(a,b,c,k);
                                  del{i}(dim1,dim2,c,k) = del{i+1}(a,b,c,k);
                              end
                          end
                      end
                  end
            elseif strcmp(lay(i).type, 'relu') == 1
                del{i} = gpuArray(zeros(size(Z{i})));
                del{i}(Z{i+1}>0) = del{i+1}(Z{i+1}>0);
            elseif strcmp(lay(i+1).type, 'conv') == 1
                del{i} = gpuArray(zeros(size(Z{i})));
                for c=1:lay(i+1).no_channel % no of input channel
                    for f=1:lay(i+1).no_filter % no of filter
                        del{i}(:,:,c,:) = (convn(del{i+1}(:,:,f),lay(i+1).w(:,:,c,f),'full'));
                    end
                end
            end
        end


        for i=nl-1:-1:1
            if strcmp(lay(i).type, 'fully') == 1
                Db{i} = Db{i} + sum(del{i+1},2); % Sum along dimension of batch size
                Dw{i} = (Dw{i} + del{i+1}*(Z{i}'));
            elseif strcmp(lay(i).type, 'relu') == 1
                %no weight update 
            elseif strcmp(lay(i).type,'maxpool') == 1
                %no weight update here
            elseif strcmp(lay(i).type,'conv') == 1
                Dw{i} = gpuArray(zeros(size(lay(i).w)));
%                 for k=1:no_b
                for c=1:lay(i).no_channel % no of input channel
                    for f=1:lay(i).no_filter % no of filter
                        Dw{i}(:,:,c,f) = sum(convn(rot90(Z{i}(:,:,c,:),2),del{i+1}(:,:,f),'valid'),4);
                    end
                end
%                 end
            end
        end
        % update the weights for the connections 
        for i=nl-1:-1:1
%             Sdw{i} = b2*Sdw{i} + (1-b2)*(Dw{i}.*Dw{i});
%             Sdb{i} = b2*Sdb{i} + (1-b2)*(Db{i}.*Db{i});
%             Vdw{i} = b1*Vdw{i} + (1-b1)*(Dw{i});
%             Vdb{i} = b1*Vdb{i} + (1-b1)*(Db{i});
%             Vdwcorrected = Vdw{i}/(1-b1^epoch);
%             Vdbcorrected = Vdb{i}/(1-b1^epoch);
%             Sdwcorrected = Sdw{i}/(1-b2^epoch);
%             Sdbcorrected = Sdb{i}/(1-b2^epoch);
            lay(i).w = lay(i).w - (eta/no_b)*(Dw{i});
            lay(i).b = lay(i).b - (eta/no_b)*(Db{i});
        end
    end

    % compute the training error
    % forward pass of the network
    trainout = forward_conv(newtrainX,lay,nl);
    
    valout = forward_conv(newvalX,lay,nl); 
      
    
    
    trainerror(epoch) = (norm(trainY-trainout).^2);
    valerror(epoch) = (norm(valY-valout).^2);
    fprintf('epoch %d: training error: %f validation error: %f \n',epoch,trainerror(epoch),valerror(epoch));
    fprintf('\t time: %f training MSE: %f validation MSE: %f',toc(tep),trainerror(epoch)/N,valerror(epoch)/valN);
   
    if any(trainerror(epoch) > trainerror(1:epoch-1))
        fprintf(' Train++ minTrain %f',min(trainerror(1:epoch-1)));
    end
    if any(valerror(epoch) > valerror(1:epoch-1))
        fprintf(' Val++ minVal %f',min(valerror(1:epoch-1)));
        
    else
        testN = size(testX,1);
    
        testZ = (cell(nl,1));
        testZ{1} = (testX');
        for i=1:nl-2
            testZ{i+1} = (p*sigmf(repmat(lay(i).b,1,testN)+lay(i).w*(testZ{i}),[1,0]));

        end
        testZ{nl} = repmat(lay(nl-1).b,1,testN)+(lay(nl-1).w*[testZ{nl-1}]);
        testZ{nl} = testZ{nl}';
        bestprediction = testZ{nl};
%         if epoch > 10
            dlmwrite((['predict/prediction',num2str(valerror(epoch)),'.txt']),bestprediction,'precision',12);
%         end
    end
    fprintf('\n');
end
%% Predict Z



testN = size(testX,1);
    
testZ = (cell(nl,1));
testZ{1} = (testX');
for i=1:nl-2
    testZ{i+1} = (p*sigmf(repmat(lay(i).b,1,testN)+lay(i).w*(testZ{i}),[1,0]));

end
testZ{nl} = repmat(lay(nl-1).b,1,testN)+(lay(nl-1).w*[testZ{nl-1}]);
testZ{nl} = testZ{nl}';
prediction = testZ{nl};
dlmwrite('prediction-test3.txt',prediction,'precision',12);


