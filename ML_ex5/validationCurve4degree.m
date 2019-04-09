function [degree_vec, error_train, error_val] = ...
    validationCurve4degree(X, y, Xval, yval)


m = size(X, 1);

% Selected values of dimension (you should not change this)
degree_vec = zeros(10,1);
for i = 1:10
    degree_vec(i) = i;
end

% You need to return these variables correctly.
error_train = zeros(length(degree_vec), 1);
error_val = zeros(length(degree_vec), 1);


% ====================== YOUR CODE HERE ======================

for p = 1:length(degree_vec)
    [X_poly] = polyFeatures(X, p);
    [X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
    X_poly = [ones(m, 1), X_poly];                   % Add Ones
    
    X_poly_val = polyFeatures(Xval, p);
    X_poly_val = bsxfun(@minus, X_poly_val, mu);
    X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
    X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

    [theta] = trainLinearReg(X_poly, y, 0);
    [error_train(p), grad_train] = linearRegCostFunction(X_poly, y, theta, 0);
    [error_val(p), grad_val] = linearRegCostFunction(X_poly_val, yval, theta, 0);
end









% =========================================================================

end
