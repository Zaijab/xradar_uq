function [t, Y] = PropagateSamples(nSamples, t0, tf, mu, P)

    options = odeset('AbsTol',1e-11,'RelTol',1e-13);
    tstep = 0.001;
    tspan = [t0 tf];
    stepCount = floor((tf-t0) / tstep);
    t = t0:tstep:tf;

    % generate random points within input PDF
    ICs = mvnrnd(mu, P, nSamples);
    
    Y0 = ICs';
    % Initialize the output matrix for the results
    Y = zeros(nSamples, 6, stepCount);
    
    
    for idx = 1:nSamples
        
        Y_init = ICs(idx, :);
        for dt = t
            [~, Y_D] = ode113(@CR3BP, [dt dt+tstep], Y_init, options);
            Y(:, idx) = Y_D(end, :);
        end
    end

end