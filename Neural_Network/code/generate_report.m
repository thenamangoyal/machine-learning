close all;
clear all;
rng('default');

% Getdata
re_read = false;
if re_read==true
   [X,Y] = gettraindata(fullfile('.','steering'),'data.txt');
   save('wholeTraindata.mat','X','Y'); % no separation in validation or train set
else
   load('wholeTraindata.mat','X','Y')
end

% Create Train and Validation
rng('default');
sep = tic; 	
totalN = size(X,1);
split = randperm(totalN);
split_n = floor((0.8)*totalN);
trainX = gpuArray(X(split(1:split_n),:));
trainY = gpuArray(Y(split(1:split_n),:));
valX = gpuArray(X(split(split_n+1:totalN),:));
valY = gpuArray(Y(split(split_n+1:totalN),:));
fprintf('Create train and validation set in %f sec\n',toc(sep));
trainN = size(trainX,1);
valN = size(valX,1);

%% Read test Data
re_read_test = false;
if re_read_test==true
   [testX] = gettestdata(fullfile('.'),'test-data.txt');
   save('wholeTestdata.mat','testX'); % no separation in validation or train set
else
   load('wholeTestdata.mat','testX');
end


%% i (5000 epochs) with a learning rate of 0.01. (no dropout, minibatch size of 64).
% Call MLP to learn
beginame = '1_';
nEpochs = 5000;
eta = 0.01; % learning rate
p = 1; % dropout - keep with p
m = 64; %minibatch size
[Trainerror, Valerror,predict1] = MLP(trainX,trainY,valX,valY,testX,nEpochs, eta, m, p);

% Save file
matpath = fullfile('.','report','mat');
figpath = fullfile('.','report','fig');
pngpath = fullfile('.','report');

report_save(Trainerror, Valerror,trainN, valN, eta, p,m,beginame, matpath, figpath, pngpath);
dlmwrite('prediction1new.txt',predict1,'precision',12);
%% ii (1000 epochs) with a fixed learning rate of 0.01 for three minibatch sizes – 32, 64, 128
beginame = '2_';
nEpochs = 1000;
eta = 0.01; % learning rate
p = 1; % dropout - keep with p
for m = [32,64,128] %minibatch size
[Trainerror, Valerror,predict2] = MLP(trainX,trainY,valX,valY,testX,nEpochs, eta, m, p);

% Save file
matpath = fullfile('.','report','mat');
figpath = fullfile('.','report','fig');
pngpath = fullfile('.','report');
dlmwrite((['prediction2new',num2str(m),'.txt']),predict2,'precision',12);
report_save(Trainerror, Valerror,trainN, valN, eta, p,m,beginame, matpath, figpath, pngpath);
end
%% iii (1000 epochs) with a learning rate of 0.001 and dropout probability of 0.5 for the first, second and third layers
beginame = '3_';
nEpochs = 1000;
eta = 0.001; % learning rate
p = 0.5; % dropout - keep with p
m = 64; %minibatch size
[Trainerror, Valerror,predict3] = MLP(trainX,trainY,valX,valY,testX,nEpochs, eta, m, p);

% Save file
matpath = fullfile('.','report','mat');
figpath = fullfile('.','report','fig');
pngpath = fullfile('.','report');
dlmwrite((['prediction3newdropout','.txt']),predict3,'precision',12);
report_save(Trainerror, Valerror,trainN, valN, eta, p,m,beginame, matpath, figpath, pngpath);

%% iv (1000 epochs) with different learning rates 0.05, 0.001, 0.005 (no drop out, minibatch size – 64)
beginame = '4_';
nEpochs = 1000;
for eta = [0.05, 0.001, 0.005] % learning rate
p = 1; % dropout - keep with p
m = 64; %minibatch size
[Trainerror, Valerror,predict4] = MLP(trainX,trainY,valX,valY,testX,nEpochs, eta, m, p);

% Save file
matpath = fullfile('.','report','mat');
figpath = fullfile('.','report','fig');
pngpath = fullfile('.','report');
dlmwrite((['prediction4new',num2str(eta),'.txt']),predict4,'precision',12);
report_save(Trainerror, Valerror,trainN, valN, eta, p,m,beginame, matpath, figpath, pngpath);
end