function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

degree = 6;
out = ones(size(X1(:,1)));  %第一列已经是1了，且只有一列。
for i = 1:degree   % i = 1 ~ 6
    for j = 0:i    % j = 0:1 ~ 0:6
        % out扩大一列，X1和X2的指数加起来等于i，总共(1+7)*7/2=28列 {i=1(2列)、i=2(3列)、i=3(4列)、i=4(5列)、i=5(6列)、i=6(7列)}
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);  
    end
end

end