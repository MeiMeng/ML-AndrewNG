function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    Predictions = X*theta;
    %外面循环一次，都要同时更新一次所有theta
    % for i = 1 : length(theta)  %matlab中的数组角标是从1开始的 
    %     theta(i) = theta(i) - alpha * ((Predictions-y)'*X(:,i))/m;   %更新的是theta几，乘的就是X的第几列。(求偏导数)
    % end
    theta = theta - alpha * (X'*(Predictions-y))/m;  %批量梯度下降,J关于各个theta的偏导数组成一个向量。(X'*(Predictions-y))是(n+1)行1列的向量。

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
 

end

end
