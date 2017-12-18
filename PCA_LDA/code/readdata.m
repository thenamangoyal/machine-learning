function [ rawX, label ] = readdata(  )

    rawX = importdata('data.txt');
    vectlabel = (importdata('label.txt'));
    N = size(vectlabel,1); % data points
%     D = size(rawX,2);
    label = zeros(N,1);
    for i=1:N
        label(i) = mod(find(vectlabel(i,:)),10);
    end

end