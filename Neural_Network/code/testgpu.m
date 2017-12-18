function [ out ] = testgpu( X,lay,nl)
    N = size(X,1);
    Z = (cell(nl,1));
    Z{1} = (X');
    for i=1:nl-2
        Z{i+1} = (sigmf(repmat(lay(i).b,1,N)+lay(i).w*(Z{i}),[1,0]));
    end
    Z{nl} = repmat(lay(nl-1).b,1,N)+(lay(nl-1).w*[Z{nl-1}]);
    Z{nl} = Z{nl}';
    out = Z{nl};
end

