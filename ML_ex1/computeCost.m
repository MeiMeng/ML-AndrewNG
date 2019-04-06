function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


predictionVector = X*theta;
E = predictionVector-y;

%每个误差算平方，然后再加起来
errorQuadraticSum = E' * E;

%代价要在误差的平方和上除以2m
J = errorQuadraticSum/(2*m);



% =========================================================================

end
