function [t, Y] = PropagateSamples(Y0, t0, tstep, tf) 
    % Takes an Nx6 matrix containing the initial state of N samples.
    % Outputs a Nx6xT matrix containing the state of N samples at time t
    % from t0:tstep:tf
    
    nSamples = size(Y0,1);
    divisionFactor = 10;
    tspan_coarse = t0:tstep:tf;                     % Output time vector
    tspan_fine = t0:tstep/divisionFactor:tf;        % Internal integration time vector
    stepCount = numel(tspan_coarse);
    
    t = tspan_coarse;
    Y = zeros(nSamples, 6, stepCount);
    
    options = odeset('AbsTol',1e-11,'RelTol',1e-13);
    
    for idx = 1:nSamples
        Y_init = Y0(idx, :).';  % Ensure it's a column vector
        [t_out, Y_out] = ode45(@CR3BP, tspan_fine, Y_init, options);
        
        % Interpolate results at the coarse time points
        Y_interp = interp1(t_out, Y_out, tspan_coarse, 'linear');
        
        if size(Y_interp, 1) ~= stepCount
            error("Time step mismatch: expected %d, got %d", stepCount, size(Y_interp, 1));
        end

        Y(idx, :, :) = permute(Y_interp, [3 2 1]);  % (1 x 6 x T)
    end
end