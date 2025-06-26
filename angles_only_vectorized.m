% Put Samples N x 6 x 1 for positions
function measurements = angles_only_vectorized(positions, mu, noise_cov)
    if nargin < 3 || isempty(noise_cov)
        noise_cov = diag([(0.5 * pi / 180)^2, (0.5 * pi / 180)^2]);
    end

    % Extract spatial dimensions: positions is N x 6 x 1 or N x 3 x 1
    x_rel = positions(:, 1, 1) + mu;
    y_rel = positions(:, 2, 1);
    z_rel = positions(:, 3, 1);

    alpha = atan2(y_rel, x_rel);
    epsilon = asin(z_rel ./ sqrt(x_rel.^2 + y_rel.^2 + z_rel.^2));
    measurements = cat(3, alpha, epsilon);  % N x 1 x 2
    measurements = permute(measurements, [1, 3, 2]);  % N x 2 x 1

    if ~islogical(noise_cov) || noise_cov
        N = size(positions, 1);
        noise = mvnrnd(zeros(1, 2), noise_cov, N);  % N x 2
        measurements = measurements + reshape(noise, [N, 2, 1]);
    end
end