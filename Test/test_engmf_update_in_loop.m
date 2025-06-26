% Test EnGMF update with actual measurements and states from main.m
clear; clc;

% Gravitational constant (same as main.m)
mu = 0.012150584269940; 

% 1. Get initial points same as main.m
nSamples = 100;
mu_initial = [1.021339954388544 -0.000000045869005 -0.181619950369762 0.000000617839352 -0.101759879771430 0.000001049698173]';
P_initial = 1.0e-08 *[ ...
   0.067741479217036  -0.000029214433641   0.000292500436172   0.000343197998120  -0.000801894296500  -0.000076851751508
  -0.000029214433641   0.067949657828148  -0.000045655889447   0.000112485276059   0.002893878948354  -0.000038999497288
   0.000292500436172  -0.000045655889447   0.067754170807105  -0.000931574297640   0.000434803811832   0.000042975146838
   0.000343197998120   0.000112485276059  -0.000931574297640   0.950650788374193   0.004879599683572   0.000839738344685
  -0.000801894296500   0.002893878948354   0.000434803811832   0.004879599683572   0.955575624017479  -0.002913896437441
  -0.000076851751508  -0.000038999497288   0.000042975146838   0.000839738344685  -0.002913896437441   0.954675354567578];

ICs = mvnrnd(mu_initial, P_initial, nSamples);

% 2. Propagate points like in main.m
measurementInterval = 0.005;
frameInterval = 0.001;
t0 = 0;      
tf = 0.75103;

[t, Y] = PropagateSamples(ICs, t0, frameInterval, measurementInterval);

% 3. Run EnGMF update on angles_only measurement
% Get final propagated states
Y_final = Y(:,:,end);  % nSamples x 6 matrix

% Measurement covariance for angles-only (azimuth, elevation)
angle_std = 0.5 * pi / 180;  % 0.5 degrees in radians
R = diag([angle_std^2, angle_std^2]);


% Create angles-only measurement (azimuth, elevation)
Z = angles_only(Y_final, mu, R, true);  % Returns measurements with noise

% Set up for EnGMF update
% Convert samples to the format expected by engmf_update
m_predict = Y_final';  % 6 x nSamples (each column is a state vector)

% Create uniform prior covariances using KDE bandwidth
state_dim = 6;
sample_mean = mean(Y_final, 1);
centered_samples = Y_final - sample_mean;
sample_cov = (centered_samples' * centered_samples) / (nSamples - 1);

% Silverman's rule for bandwidth
beta_silv = (4 / (nSamples * (state_dim + 2)))^(2 / (state_dim + 4));
B = beta_silv * sample_cov;

% Create covariance array (6 x 6 x nSamples)
P_predict = repmat(B, [1, 1, nSamples]);

% Measurement covariance for angles-only (azimuth, elevation)
angle_std = 0.5 * pi / 180;  % 0.5 degrees in radians
R = diag([angle_std^2, angle_std^2]);

% Set up configuration and model for measurement function
cfig = Config();
model.rIs = zeros(3,1);  % sensor position
model.vIs = zeros(3,1);  % sensor velocity

% Run EnGMF update for each measurement
num_measurements = size(Z, 1);
weights_out = cell(num_measurements, 1);
means_out = cell(num_measurements, 1);
covs_out = cell(num_measurements, 1);

for i = 1:num_measurements
    y = Z(i, :)';  % Current measurement [azimuth; elevation]
    
    % Run the EKF-style update for each particle (same as run_filter.m)
    [m_update, P_update, U_update] = update(cfig, model, y, R, m_predict, P_predict);
    
    % Compute weights using log-sum-exp for numerical stability
    log_weights = U_update - max(U_update);
    weights = exp(log_weights);
    weights = weights / sum(weights);  % Normalize
    
    % Store results
    weights_out{i} = weights;
    means_out{i} = m_update;
    covs_out{i} = P_update;
    
    fprintf('Measurement %d: Updated %d components, total weight = %.4f\n', ...
            i, length(weights), sum(weights));
end

% Display summary
fprintf('\nEnGMF Update Complete:\n');
fprintf('- Processed %d measurements\n', num_measurements);
fprintf('- Each update returned %d weighted components\n', nSamples);
fprintf('- Final mean position: [%.6f, %.6f, %.6f]\n', ...
        mean(means_out{end}(1:3,:), 2));