function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Calculating unregularized cost function:
XX = [ones(m,1) X];
for i=1:m
  Xnew = XX(i,:)';
  z2 = Theta1 * Xnew;
  a2 = sigmoid(z2);
  a2 = [1;a2];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3); %h(x)
  for k=1:num_labels
    yk = 0;
    if (y(i,1) == k)
      yk = 1;
    endif
    J += -(yk*log(a3(k)))-((1-yk)*log(1-(a3(k))));
  end
  
end
J = J / m;
%t1 = [ones(size(Theta1,1),1) Theta1];
sum = 0;
for j = 1:size(Theta1,1)
  for k=2:size(Theta1,2)
    sum = sum + (Theta1(j,k) * Theta1(j,k));
  end
end
for j = 1:size(Theta2,1)
  for k=2:size(Theta2,2)
    sum = sum + (Theta2(j,k) * Theta2(j,k));
  end
end
sum = (sum * lambda) / (2 * m);
J = J + sum;

% Back Propagation
del1 = zeros(size(Theta1,1),size(Theta1,2));
del2 = zeros(size(Theta2,1),size(Theta2,2));
for t=1:m
  a1 = X(t,:);
  a1 = a1';
  a1 = [1;a1];
  z2 = Theta1 * a1;
  a2 = sigmoid(z2);
  a2 = [1;a2];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  delta_3 = zeros(num_labels,1);
  for k = 1:num_labels
    yk = 0;
    if (k == y(t,1))
      yk = 1;
    endif
    delta_3(k,1) = a3(k,1) - yk;
  end
  delta_2 = zeros(size(z2,1),1);
  %for l=1:size(Theta2,1)
    p = (Theta2)';
    %p(1,:)=[];
    zz2=[1;z2];
    delta_2 = (p * delta_3) .* sigmoidGradient(zz2); 
  %end
  delta_2 = delta_2(2:end);
  del1 = del1 + (delta_2 * (a1)');
  del2 = del2 + (delta_3 * (a2)');
  
end
del1 = del1 ./ m;
del2 = del2 ./ m;
Theta1_grad = del1;
Theta2_grad = del2;

% regularization
temp1 = Theta1_grad(:,1);
temp2 = Theta2_grad(:,1);
tmp_t1 = Theta1;
tmp_t2 = Theta2;
tmp_t1 = (lambda / m) * tmp_t1;
tmp_t2 = (lambda / m) * tmp_t2;
Theta1_grad = Theta1_grad .+ tmp_t1;
Theta2_grad = Theta2_grad .+ tmp_t2;
Theta1_grad(:,[1])=[];
Theta1_grad = [temp1,Theta1_grad];
Theta2_grad(:,[1])=[];
Theta2_grad = [temp2,Theta2_grad];
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
