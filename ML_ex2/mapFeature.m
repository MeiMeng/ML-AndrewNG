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
out = ones(size(X1(:,1)));  %��һ���Ѿ���1�ˣ���ֻ��һ�С�
for i = 1:degree   % i = 1 ~ 6
    for j = 0:i    % j = 0:1 ~ 0:6
        % out����һ�У�X1��X2��ָ������������i���ܹ�(1+7)*7/2=28�� {i=1(2��)��i=2(3��)��i=3(4��)��i=4(5��)��i=5(6��)��i=6(7��)}
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);  
    end
end

end