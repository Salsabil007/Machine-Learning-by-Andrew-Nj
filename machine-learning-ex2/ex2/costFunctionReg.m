function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
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
for j=1:size(grad)
  p = h - y;
  q = X(:,j);
  d = p .* q;
  if (j==1)
    grad(j) = sum(d)/m;
  elseif
    grad(j) = sum(d)/m + ((lambda/m) * theta(j));
  endif
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
