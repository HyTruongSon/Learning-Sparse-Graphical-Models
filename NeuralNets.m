function [predict] = NeuralNets(W0, W1, X)
    predict = ReLU(ReLU(X * W0) * W1);
end