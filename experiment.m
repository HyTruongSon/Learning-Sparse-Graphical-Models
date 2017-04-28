function [] = experiment()
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
    
    [predict_train, accuracy_train] = NeuralNets(W0, W1, X_train, Y_train);
    fprintf('Accuracy on the train set: %.2f\n', accuracy_train);
    
    [predict_test, accuracy_test] = NeuralNets(W0, W1, X_test, Y_test);
    fprintf('Accuracy on the test set: %.2f\n', accuracy_test);
    
    expected = reshape(Y_test(1, 1:end), 10, 10);
    predict = reshape(predict_test(1, 1:end), 10, 10);
    subplot(1, 2, 1);
    imagesc(predict);
    title('Predicted');
    subplot(1, 2, 2);
    imagesc(expected);
    title('Expected');
end