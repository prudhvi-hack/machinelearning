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

                 
fTheta1=Theta1;
fTheta2=Theta2;
% Setup some useful variables
m = size(X, 1);
X=[ones(m,1),X];        
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

temp=X*Theta1';


a=sigmoid(temp);

a=[ones(m,1) a];

temp2=a*Theta2';

a2=sigmoid(temp2);

for c=1:num_labels
  
 

J(c)=((y==c)'*(log(a2))(:,c:c)+(1-(y==c))'*(log(1-a2))(:,c:c));


end

J=sum((-1/m)*J');






Theta1=Theta1(:,2:(input_layer_size+1));

Theta2=Theta2(:,2:(hidden_layer_size+1));

Theta1=Theta1.^2;

Theta2=Theta2.^2;

tem=sum(sum(Theta1));

tem=tem+sum(sum(Theta2));


J=J+((tem)*lambda)/(2*m);





delta1=0;

delta2=0;
delta3=0;
for t=1:m;
  
  
  a_1=(X(t:t,:))';
  
  z_1=a_1(2:end);
  
  
  z_2=(a_1'*fTheta1')';
  
  a_2=sigmoid(z_2);
  z_2=[1;z_2];
  
  a_2=[1;a_2];
  
  z_3=(a_2'*fTheta2')';
  
  
  a_3=sigmoid(z_3);
  
  
  d_3=a_3-([1;2;3;4;5;6;7;8;9;0](1:num_labels)==y(t));
  
  
  
  
  
  d_2=(fTheta2'*d_3).*sigmoidGradient(z_2);
  
  d_2=d_2(2:end);
  
  delta1=delta1+d_2*a_1';
  
  delta2=delta2+d_3*a_2';
  
  
end

delta1=delta1/m;

delta2=delta2/m;


delta1=delta1+(lambda/m)*[[0;zeros(size(fTheta1)-1)(:,1:1)] fTheta1(:,2:(input_layer_size+1))];

delta2=delta2+(lambda/m)*[[0;zeros(size(fTheta2)-1)(:,1:1)] fTheta2(:,2:(hidden_layer_size+1))];






grad=[delta1(:) ; delta2(:)];






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
%grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
