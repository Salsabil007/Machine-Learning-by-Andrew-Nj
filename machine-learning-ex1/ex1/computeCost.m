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

t1 = X(:,2);
t1 = t1';
t2 = ones(1,m);
p = [t2;t1];
thetanew=theta';
pnew=thetanew * p;
pnew = pnew';
pnew = pnew - y;
pnew = pnew.^2;
J = (0.5 / m) * sum(pnew);



% =========================================================================

end
