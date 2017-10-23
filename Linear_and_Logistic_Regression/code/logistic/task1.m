function [ output_args ] = task1( )
    load('credit.mat');
    plot(data(find(label==1),1),data(find(label==1),2),'b+',data(find(label==0),1),data(find(label==0),2),'ro'),
    legend('Positive','Negative');
end
