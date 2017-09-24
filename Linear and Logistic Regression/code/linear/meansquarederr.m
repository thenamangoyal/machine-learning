function [X] = meansquarederr(T,Tdash)
    X=T-Tdash;
    X=(X')*X/(2*size(T,1));
end