function [ rawprojX, U, transX ] = runpca( rawX , elim)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    N = size(rawX,1); % no of data points
    D = size(rawX,2);
    meanX = mean(rawX);
    centX = rawX - repmat(meanX,N,1);
%     %%
%     covar = zeros(D,D);
%     for i=1:N
%         covar = covar + X(i,:)'*X(i,:);
%     end

    covar = (1/N)*(centX'*centX);
    [tu,ts,tv] = svd(covar); % u*s*w'
    error = inf;
    to_pick =0;
    while error > elim
        to_pick = to_pick+1;

        U = tv(:,1:to_pick);

        transX = centX*U;
        projX = transX*U';
        rawprojX = projX+repmat(meanX,N,1);
        error = norm(rawX - rawprojX,'fro')^2/N;
    %     error = sum(sum((rawX - rawprojX).*(rawX - rawprojX),2))/N;

    end
end