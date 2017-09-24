function [ output_args ] = mylinridgereg( X, Y, lambda )
    I = eye(size(X,2));
    if size(I,1) >= 1 && size(I,2) >= 1
        I(1,1) = 0;
    end
    output_args = pinv((X')*X + lambda*I)*(X')*Y;
end

