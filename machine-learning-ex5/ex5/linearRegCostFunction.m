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
% Calculating J
hx = theta' * X';
hx = hx' - y;
hx = hx .^ 2;
J = sum(hx)/(2 * m);
temp = theta;
temp(1,1) = 0;
temp = temp .^ 2;
J = J + ((sum(temp) * lambda)/(2*m));

% Calculating grad
h = theta' * X';
h = h';
grad =  X' * (h-y);
grad = grad ./ m;
temp1 = grad(1);
temp2 = theta;
temp2 = temp2 .* (lambda/m);
grad = grad .+ temp2;
grad(1)=temp1;














% =========================================================================

grad = grad(:);

end
