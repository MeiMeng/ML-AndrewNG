function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

pos = find(y==1);  % y=1的索引值
neg = find(y==0);  % y=0的索引值

scatter( X(pos,1), X(pos,2),  30, 'k+',  'LineWidth', 2);
scatter( X(neg,1), X(neg,2),  30, 'ko',  'MarkerFaceColor', 'y');
  
% =========================================================================

hold off;

end
