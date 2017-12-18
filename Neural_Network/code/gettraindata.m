function [X,Y] = gettraindata(path,data_file_name)
fprintf('Reading Train Data\n');
read_time = tic;
F = importdata(fullfile(path,data_file_name));
imglist = fullfile(path,F.textdata);

anglelist = F.data;
img1 = [];
found = false;
totake= [];
no_img = length(imglist);
for i=1:no_img
    if exist(char(imglist(i)),'file') == 2
        if found == false
            img1 = im2double(rgb2gray(imread(char(imglist(i)))));        
            found = true;            
        end
        totake = [totake i];
    end
end
imglist = imglist(totake,:);
anglelist = anglelist(totake,:);
X = (zeros(length(imglist),numel(img1)));
Y = (zeros(length(imglist),1));
no_img = length(imglist);
for i=1:no_img
    temp = im2double(rgb2gray(imread(char(imglist(i,:)))));
    X(i,:) = temp(:);
    Y(i) = anglelist(i);
end
fprintf('Train Read Data in %f s\n',toc(read_time));
end