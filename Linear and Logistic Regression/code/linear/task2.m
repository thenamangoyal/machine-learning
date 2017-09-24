function [ output_args, mean_data, standard_deviation_data ] = task2( input_args )
    mean_data = mean(input_args.X);
    standard_deviation_data = std(input_args.X ,1);
    input_args.X = (input_args.X - mean_data)./standard_deviation_data;
    input_args.X = [ones(size(input_args.X,1),1) input_args.X];
    output_args = input_args;
end

