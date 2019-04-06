function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);  % 5000
num_labels = size(Theta2, 1); % 10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);  % 5000*1

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%输入层
X = [ones(m,1) X];

%隐藏层
% 25*1 = 25*401 * 401*1
A = sigmoid(Theta1 * X');  % 25*5000
A = [ones(m,1) A'];  % 5000*26

%输出层
% 10*1 = 10*26 * 26*1
H = sigmoid(Theta2 * A');  % 10*5000
H = H';  % 5000*10

[M,p] = max(H, [], 2);  %选出每一行中最大的那个
% M，维数m*1，是每行最大值；
% p，维数m*1，是每行最大值的列号

end
