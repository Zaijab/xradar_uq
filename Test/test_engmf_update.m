% Example: 2D position state with range measurement
clc; clear all; close all; 


% State: [x; y], Measurement: range = sqrt(x^2 + y^2)

% Define measurement function and Jacobian
h_func = @(model, x, flag) sqrt(x(1)^2 + x(2)^2);
H_func = @(model, x) [x(1)/sqrt(x(1)^2 + x(2)^2), x(2)/sqrt(x(1)^2 + x(2)^2)];

% Prior GM parameters (3 components)
m_predict = [1, -1, 2; 2, 1, -1];              % 2x3 means
P_predict = repmat(eye(2) * 0.5, [1,1,3]);     % 2x2x3 covariances  
w_predict = [0.4; 0.3; 0.3];                   % 3x1 weights

% Measurement
z = 1.5;                    % Range measurement
R = 0.1;                    % Measurement variance
J_rsp = 50;                 % Resampling count

% Function call
[m_update, P_update, w_update] = engmf_update(h_func, H_func, z, ...
                                 m_predict, P_predict, w_predict, R, J_rsp);
                                 
% Output shapes:
% m_update: 2x50, P_update: 2x2x50, w_update: 50x1
w_update
