function [X] = gettestdata(path,data_file_name)
fprintf('Reading Test Data\n');
read_time = tic;
imglist = fullfile(path,importdata(fullfile(path,data_file_name)));

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
new_imglist = imglist(totake,:);
X = (zeros(length(new_imglist),numel(img1)));
no_img = length(new_imglist);
for i=1:no_img
    temp = im2double(rgb2gray(imread(char(new_imglist(i,:)))));
    X(i,:) = temp(:);
end

fprintf('Test Data Read in %f s\n',toc(read_time));
end