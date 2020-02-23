function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

for i=1:num_movies
  for j=1:num_users
    if (R(i,j)==1)
      theta_jt = Theta(j,:);
      p = (theta_jt * (X(i,:))') - Y(i,j);
      p = p * p;
      J = J + p;
    endif
  end
end
J = J / 2;

%regularized cost function
reg=0;
for j=1:num_users
  for k=1:num_features
    reg = reg + (Theta(j,k)) ^ 2;
  end
end
J = J + (reg * lambda)/2;
reg=0;
for i=1:num_movies
  for k=1:num_features
    reg = reg + (X(i,k)) ^ 2;
  end
end
J = J + (reg * lambda)/2;

for i=1:num_movies
  for k=1:num_features
    sum = 0;
    for j=1:num_users
      if (R(i,j)==1)
        theta_jt = Theta(j,:);
        p = (theta_jt * (X(i,:))') - Y(i,j);
        p = (p * Theta(j,k));
        sum = sum + p;
      endif
    end
    X_grad(i,k) = sum + (lambda * X(i,k));
  end
end
for j=1:num_users
  for k=1:num_features
    sum = 0;
    for i=1:num_movies
      if (R(i,j)==1)
        theta_jt = Theta(j,:);
        p = (theta_jt * (X(i,:))') - Y(i,j);
        p = p * X(i,k);
        sum = sum + p;
      endif
    end
    Theta_grad(j,k) = sum + (lambda * Theta(j,k));
  end
end
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
















% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
