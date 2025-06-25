function [Y0, Y] = PropagateInitialPDF(nSamples, t0, tf, mu, P)

    options = odeset('AbsTol',1e-11,'RelTol',1e-13);
    tstep = 0.001;
    tspan = [t0 tf];

    % generate random points within input PDF
    ICs = mvnrnd(mu, P, nSamples);
    
    Y0 = ICs';
    % Initialize the output matrix for the results
    Y = zeros(6, nSamples);
    
    
    for idx = 1:nSamples
         
        X_init = ICs(idx, :);
    
        [t, X_D] = ode113(@CR3BP, tspan, X_init, options);
        Y(:, idx) = X_D(end, :)';
    end

end