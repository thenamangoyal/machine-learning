function [ output_args ] = featuretransform(X,degree)
% X contains 2 features and 1 valued constant feature
    Xnew = [X(:,1)];
    for i=0:degree
        for j=0:degree-i
            if (i~=0) || (j~=0)
                Xnew = [Xnew (X(:,2).^j).*(X(:,3).^i)];
            end
        end
    end
    output_args = Xnew;
end
