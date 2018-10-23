function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values

m = length(y); % number of training examples

%n = length(theta); % number of features

J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % attempt 0 (not vectorized)
    
%     tmplist0 = nan(m,1);
%     tmplist1 = nan(m,1);
%     
%     for i=1:m
%         tmplist0(i) = theta'*X(i,:)'-y(i);
%         tmplist1(i) = (theta'*X(i,:)'-y(i))*X(i,2);
%     end
%         
%     theta(1) = theta(1) - alpha * nanmean(tmplist0);
%     theta(2) = theta(2) - alpha * nanmean(tmplist1);

%attempt 1
% 
% delta = (1/m)*(sum((X*theta-y)*X));
% 
% theta = theta - alpha*delta;



% attempt 2 (with help from tutorial)

 h = X*theta; errors = h - y; theta_change = (1/m)*alpha*(X'*errors); theta = theta - theta_change; 


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
