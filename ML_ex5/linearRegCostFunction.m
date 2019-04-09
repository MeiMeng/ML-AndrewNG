function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));  % n*1

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% X=m*n  theta=n*1
H = X*theta;  % m*1
E = H-y;      % m*1

theta_reg = [0; theta(2:end)];  
J = (1/(2*m)) * (E'*E + lambda*(theta_reg'*theta_reg));

grad = (1/m) * (X'*E + lambda*theta_reg);  % n*1

% =========================================================================

grad = grad(:);

end
