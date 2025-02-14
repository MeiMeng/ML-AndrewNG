function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    Predictions = X*theta;
    E = Predictions - y;
    %外面循环一次，都要同时更新一次所有theta
%     for i = 1 : length(theta)  %matlab中的数组角标是从1开始的
%         D = E.*X(:,i);  %更新的是theta几，乘的就是X的第几列。(求偏导数)
%         theta(i) = theta(i) - alpha*(1/m)*sum(D);
%     end

    %批量梯度下降，总共有(n+1)个theta，J关于各个theta的偏导数组成一个(n+1)维向量：(X'*E) 
    theta = theta - alpha * (X'*E)/m;  %alpha是学习率 

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
