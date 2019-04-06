function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

H = sigmoid(X*theta);  %����ģ�ͺ���
reg_theta = [0; theta(2:end)];

J = (-1/m) * (y'*log(H) + (1-y)'*log(1-H)) + lambda/(2*m) * (reg_theta'*reg_theta);  %����������Ĵ���
%J = -(1/m) * sum(y.*log(H)+(1-y).*log(1-H));  %����
%J = 1/m*(-1*y'*log(sigmoid(X*theta)) - (ones(1,m)-y')*log(ones(m,1)-sigmoid(X*theta))) + lambda/(2*m) * (theta(2:end,:))'*theta(2:end,:);

E = H-y;  %���
% for i = 1 : size(theta, 1)  %����X����,Ҳ�ɱ���theta���� (��ģ����thetaֻ��1��)
%     % grad(i, 1) = (1/m) * sum(E .* X(:, i)) + (lambda/m)*reg_theta(i,1);  %J��theta��ƫ��
%     grad(i, 1) = ( X(:,i)'*E )/m + (lambda/m)*reg_theta(i);
% end

grad = ( X'*E )./m + (lambda/m).*reg_theta;

% =============================================================

end
