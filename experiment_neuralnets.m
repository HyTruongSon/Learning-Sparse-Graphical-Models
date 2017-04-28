function [] = experiment_neuralnets()
    %% Load Neural Nets
    W0 = load('saves/model-ReLU-nHidden-128-Epochs-10-LearningRate-0.01-Layer-0.dat');
    W1 = load('saves/model-ReLU-nHidden-128-Epochs-10-LearningRate-0.01-Layer-1.dat');

    %% Load data
    load('train_input.mat', 'input');
    X_train = input;
    load('train_target.mat', 'target');
    Y_train = target;
    load('test_input.mat', 'input');
    X_test = input;
    load('test_target.mat', 'target');
    Y_test = target;
    
    dim = int32(sqrt(size(Y_train, 2)));
    
    %% Evaluate on the training set
    fprintf('Evaluate on the training set\n');
    
    [predict_train] = NeuralNets(W0, W1, X_train);
    
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
    
    %% Evaluate on the testing set
    fprintf('Evaluate on the testing set\n');
    
    [predict_test] = NeuralNets(W0, W1, X_test);
    
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
    
    fprintf('Accuracy on the test set: %.2f\n', accuracy_test);
    
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