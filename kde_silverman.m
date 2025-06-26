% Example
% gm = gmdistribution([1,2; 0,0], cat(3, eye(2), eye(2)))
% points = random(gm, 10)
% new_gm = kde_silverman(points)
% new_gm.pdf([1,2])

function gm = kde_silverman(points)
    %
    
    [n, d] = size(points);  % n points in d dimensions
    assert(n > 1, 'Need multiple points');
    
    % Silverman's rule of thumb
    beta = (4/(n*(d+2)))^(2/(d+4));
    sample_cov = cov(points);
    P = beta * sample_cov;
    
    % Each point becomes a component with uniform weight
    mu = points;  % points is already n x d
    sigma = repmat(P, [1 1 n]);
    weights = ones(1,n) / n;
    
    gm = gmdistribution(mu, sigma, weights);
end