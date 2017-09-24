function [ output_args ] = task1( )
    rin = importdata('linregdata');
    rin = rmfield(rin,'rowheaders');
    rin.textdata = cell2mat(rin.textdata);
    s = size(rin.textdata);
    tdata=zeros(s(1),3);
    for i=1:s(1)
        if rin.textdata(i,1)=='F'
            tdata(i,1)=1;
        elseif rin.textdata(i,1)=='I'
            tdata(i,2)=1;
        elseif rin.textdata(i,1)=='M'
            tdata(i,3)=1;
        end
    end
   rin.textdata=tdata;
   output_args.X = [rin.textdata rin.data(:,1:end-1)];   
   output_args.Y = rin.data(:,end);
end

