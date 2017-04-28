function [predict, accuracy] = NeuralNets(W0, W1, X, Y)
    predict = ReLU(ReLU(X * W0) * W1);
    predict(predict > 0) = 1;
    accuracy = sum(sum(predict == Y)) / (size(Y, 1) * size(Y, 2)) * 100.0;
end