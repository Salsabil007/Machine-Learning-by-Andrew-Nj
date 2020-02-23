function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
J = 0;
z = X * theta;
h = sigmoid(z);
y1 = log(h);
y1 = y .* y1;
y1 = y1 * (-1);
y2 = 1 .- h;
y2 = log(y2);
y2 = (1 .- y) .* y2;
y2 = y2 * (-1);
thetanew=theta;
thetanew(1)=0;
J = (sum(y1)+sum(y2))/m + ((lambda/(2*m)) * sum(thetanew .^ 2));
grad = zeros(size(theta));

grad = (1/m) * X' * (h-y);
temp1 = grad(1);
temp2 = theta;
temp2 = temp2 .* (lambda/m);
grad = grad .+ temp2;
grad(1)=temp1;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================

grad = grad(:);

end
