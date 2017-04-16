function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


predictions = X * theta;
sqrErrors = (predictions-y).^2;
t1 = 1/(2*m) * sum(sqrErrors);
t2 = theta .^ 2;
t3 = lambda/(2*m) * (sum(t2)-theta(1)^2);
J = t1 + t3;

tt1 = 1/m * X'* (predictions - y);
tt2 = theta .* (lambda/m);
tt3 = tt1 + tt2;
tt3(1) = tt3(1) - (lambda/m)*theta(1);
grad = tt3;



% =========================================================================

grad = grad(:);

end
