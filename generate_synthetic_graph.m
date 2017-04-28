function [theta, L] = generate_synthetic_graph(dim, sparsity)
    L = rand(dim, dim);
    
    % Lower-diagonal matrix
    for i = 1 : dim
        for j = i + 1 : dim
            L(i, j) = 0;
        end
    end
    
    % Sparsifying
    for i = 1 : dim
        for j = 1 : i - 1
            r = rand();
            if r < sparsity
                L(i, j) = 0;
            end
        end
    end
    
    theta = L * L';
end