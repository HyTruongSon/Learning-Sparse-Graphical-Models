function [] = generate_synthetic_data(input_fn, target_fn, nExamples, dim, sparsity)
    nSamples = 10000;

    input_file = fopen([input_fn, '.dat'], 'w');
    target_file = fopen([target_fn, '.dat'], 'w');
    
    fprintf(input_file, '%d %d\n', nExamples, dim * dim);
    fprintf(target_file, '%d %d\n', nExamples, dim * dim);
    
    input = [];
    target = [];
    
    for example = 1 : nExamples
        % Generate a sparse graph by a precision matrix
        theta = generate_synthetic_graph(dim, sparsity);
        
        % Compute the corresponding covariance matrix
        sigma = theta^-1;
        
        % Normalize the covariance matrix (and the precision matrix accordingly)
        MAX = max(max(sigma));
        sigma = sigma / MAX;
        theta = theta * MAX;
        
        % Random samples from normal distribution with mean zero and the covariance matrix
        X = mvnrnd(zeros(dim, 1), sigma, nSamples);
        
        % Estimate the covariance matrix from the generated samples
        sigma_hat = X' * X ./ nSamples;
        
        % Input for the Neural Network
        inp = reshape(sigma_hat, 1, dim * dim);
        
        % Target for the Neural Network to learn
        out = reshape(theta, 1, dim * dim);
        out(out > 0) = 1;
        
        % Save the training example to files
        fprintf(input_file, '%.6f ', inp);
        fprintf(input_file, '\n');
        
        fprintf(target_file, '%d ', out);
        fprintf(target_file, '\n');
        
        if example == 1
            input = inp;
            target = out;
        else
            input = [input; inp];
            target = [target; out];
        end
        
        if mod(example, 1000) == 0
            fprintf('    Done generating %d examples\n', example);
        end
    end
    
    fclose(input_file);
    fclose(target_file);
    
    save([input_fn, '.mat'], 'input');
    save([target_fn, '.mat'], 'target');
end