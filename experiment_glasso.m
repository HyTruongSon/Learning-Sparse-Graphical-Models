function [] = experiment_glasso()
    %% Load data
    load('train_input.mat', 'input');
    X_train = input;
    load('train_target.mat', 'target');
    Y_train = target;
    load('test_input.mat', 'input');
    X_test = input;
    load('test_target.mat', 'target');
    Y_test = target;
    
    dim = int32(sqrt(size(Y_test, 2)));
    
    %% Graph LASSO    
    function [gradient] = Jacobian(theta, sigma_hat, lambda)
        gradient = -(theta + 1e-5 * eye(size(theta)))^-1 + sigma_hat' + lambda * sign(theta);
    end
    
    function [theta] = GLASSO(sigma_hat, learning_rate, nIterations, lambda)
        % Reshape sigma_hat to be square size
        sigma_hat = reshape(sigma_hat, dim, dim);
        
        % Initialize theta
        theta = (sigma_hat + 1e-5 * eye(size(sigma_hat)))^-1;
        
        % Gradient descent
        for iter = 1 : nIterations
            gradient = Jacobian(theta, sigma_hat, lambda);
            theta = theta - learning_rate * gradient;
        end
        
        theta = reshape(theta, 1, dim * dim);
    end

    %% Constants
    learning_rate = 0.001;
    nIterations = 20;
    lambda = 0.1;

    %% Evaluate on the training set
    fprintf('GLASSO on the training set\n');
    
    nTraining = size(X_train, 1);
    predict_train = zeros(size(Y_train));
    
    for sample = 1 : nTraining
        predict_train(sample, :) = GLASSO(X_train(sample, :), learning_rate, nIterations, lambda);
    end
    
    fprintf('Evaluate on the training set\n');
    for sample = 1 : size(X_train, 1)
        expect = reshape(Y_train(sample, :), dim, dim);
        predict = reshape(predict_train(sample, :), dim, dim);
        for i = 1 : dim
            k = sum(expect(i, :));
            [~, order] = sort(abs(predict(i, :)), 'descend');
            predict(i, order(1:k)) = 1;
            predict(i, order(k+1:end)) = 0;
        end
        predict_train(sample, :) = reshape(predict, 1, dim * dim);
    end
    
    accuracy_train = sum(sum(predict_train == Y_train)) / (size(Y_train, 1) * size(Y_train, 2)) * 100.0;
    fprintf('Accuracy on the train set: %.2f\n', accuracy_train);
    
    %% Evaluate on testing set
    fprintf('GLASSO on the testing set\n');
    
    nTesting = size(X_test, 1);
    predict_test = zeros(size(Y_test));
    
    for sample = 1 : nTesting
        predict_test(sample, :) = GLASSO(X_test(sample, :), learning_rate, nIterations, lambda);
    end
    
    fprintf('Evaluate on the testing set\n');
    for sample = 1 : size(X_test, 1)
        expect = reshape(Y_test(sample, :), dim, dim);
        predict = reshape(predict_test(sample, :), dim, dim);
        for i = 1 : dim
            k = sum(expect(i, :));
            [~, order] = sort(abs(predict(i, :)), 'descend');
            predict(i, order(1:k)) = 1;
            predict(i, order(k+1:end)) = 0;
        end
        predict_test(sample, :) = reshape(predict, 1, dim * dim);
    end
    
    accuracy_test = sum(sum(predict_test == Y_test)) / (size(Y_test, 1) * size(Y_test, 2)) * 100.0;
    fprintf('Accuracy on the train set: %.2f\n', accuracy_test);
    
    %% Visualization
    expected = reshape(Y_test(1, 1:end), 10, 10);
    predict = reshape(predict_test(1, 1:end), 10, 10);
    subplot(1, 2, 1);
    imagesc(predict);
    title('Predicted');
    subplot(1, 2, 2);
    imagesc(expected);
    title('Expected');
end