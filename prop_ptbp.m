function [wk,mk,Pk] = prop_ptbp(tkm1,tk,wkm1,mkm1,Pkm1,dH,GM,ex,mu)

%-------------------------------------------------------------------------
% Inputs:
%    tkm1 - initial time to start propagation, scalar
%    tk   - final time to end propagation, scalar
%    wkm1 - set of GM weights at time tkm1, cell array (nC x 1) with scalar
%           elements
%    mkm1 - set of GM means at time tkm1, cell array (nC x 1) with vector
%           elements (nX x 1)
%    Pkm1 - set of GM covariances at time tkm1, cell array (nC x 1) with
%           matrix elements (nX x nX)
%    dH   - entropy tolerance used to detect the effects of nonlinearity,
%           scalar
%    GM   - 
%    ex   - 
%    mu   - cislunar mass ratio
% Outputs:
%    wk   - set of GM weights at time tk, cell array (nC x 1) with scalar
%           elements
%    mk   - set of GM means at time tk, cell array (nC x 1) with vector
%           elements (nX x 1)
%    Pk   - set of GM covariances at time tk, cell array (nC x 1) with
%           matrix elements (nX x nX)
%    NOTE: nC in the outputs is greater than or equal to nC in the inputs
%-------------------------------------------------------------------------
disp(['Running adaptive GMM propagator... t0 = ', num2str(tkm1), ', tf = ', num2str(tk)])
% set parameters of the splitting algorithm
nsplit = 3; % number of components to split into

% specify integrator options including the events function which detects
% nonlinearity and stops the integration in order to apply the splitting
% algorithm
options = odeset('AbsTol',1e-12,'RelTol',1e-12,'Events',@KLdivstop);


nC = 1; % number of components at tkm1
nX = 6; % size of the state
nS   = 2*nX;                    % number of sigma points
wUKF = 0.5/nX*ones(2*nX,1);     % associated weights

% set the initial time, weights, means, and covariances
tlm1 = tkm1;
wlm1 = wkm1;
mlm1 = mkm1;
Plm1 = Pkm1;
mEKF = mkm1;
PEKF = Pkm1;

% enter while loop to continue propagating until tk is reached
while(true)
    % set up storage for integration array and differential entropy
    zlm1 = zeros(nX*(nS+nX+1)+1,nC);
    Hlm1 = zeros(nC,1);
    
    % loop over number of components
    for i = 1:nC
        % compute the sigma points for the ith component
        Slm1 = chol(Plm1{i})';
        Xlm1 = repmat(mlm1{i},1,nS) + sqrt(nX)*[+Slm1,-Slm1];
        
        % store the weight and sigma points for the ith component in the
        % integration vector
        zlm1(:,i) = [wlm1{i};reshape(Xlm1,nX*nS,1);mEKF{i};reshape(PEKF{i},nX*nX,1)];
        
        % compute the differential entropy for the ith component
        % Hlm1(i)   = 0.5*log(det(2*pi*exp(1)*Plm1{i}));
    end
    % reshape the integration array into an integration vector
    zlm1 = reshape(zlm1,(nX*(nS+nX+1)+1)*nC,1);
 
    % call ode45 to try and integrate from tlm1 to tk
    [tt,zt,~,~,ie] = ode45(@crtbp_filters_prop,[tlm1 tk],zlm1,options,GM,nC,nX,nS,wUKF,Hlm1,dH,mu);
    
    % determine if the events function triggered a stop
    if(~isempty(ie))
        % if a stop was triggered, extract the integration vector and time
        % from the last successful integration step before the stop
        tl = tt(end-1);
        zl = zt(end-1,:)';
    else
        % if a stop was not triggered, extract the integration vector and
        % time at the final integration step
        tl = tt(end);
        zl = zt(end,:)';
    end

    
    % reshape the extracted integration vector into an integration array
    zl = reshape(zl,nX*(nS+nX+1)+1,nC);
    
    % set up cell storage for the weights, means, and covariances to be
    % computed from the extracted integration vector
    wl = cell(nC,1);
    ml = cell(nC,1);
    Pl = cell(nC,1);
    mEKF = cell(nC,1);
    PEKF = cell(nC,1);
    % loop over the number of components
    for i = 1:nC
        % determine the weight
        wl{i}         = zl(1,i);
        % determine the propagated sigma points
        Xl            = reshape(zl(2:nX*nS+1,i),nX,nS);
        % compute the mean and covariance from the propagated sigma points
        ml{i} = zeros(nX,1);
        Pl{i} = zeros(nX,nX);
        for iii = 1:nS
            ml{i} = ml{i} + wUKF(iii)*Xl(:,iii);
            Pl{i} = Pl{i} + wUKF(iii)*(Xl(:,iii)*Xl(:,iii)');
        end
        Pl{i} = Pl{i} - ml{i}*ml{i}';

        
        % compute EKF est and cov
        mEKF{i} = zl(2+nS*nX:1+(nS+1)*nX,i);
        PEKF{i} = reshape(zl(2+(nS+1)*nX:1+(nX+nS+1)*nX,i),nX,nX);
    end
    
    % determine if the events function triggered a stop
    if(~isempty(ie))
        % figure out which components need to be split, i.e. which
        % components triggered the stop
        ir    = ie;
        % figure out which components are not being split
        %nr    = find(1:nC ~= ie);
        nr     = (1:nC)';
        nr(ie) = [];
        % set the new weights, means, and covariances of the elements which
        % are not being split
        wlnew = wl(nr);
        mlnew = ml(nr);
        Plnew = Pl(nr);
        mEKF = mEKF(nr);
        PEKF = PEKF(nr);
        % loop over the number of components which are being split
        for i = 1:length(ir)
            % apply the splitting algorithm to the ith element of the set
            % of components being split
            [wls,mls,Pls] = gausssplit(wl{ir(i)},ml{ir(i)},Pl{ir(i)},GM,nsplit,ex,tl,mu);
            % accumulate the split weights, means, and covariances to the
            % components which did not get split
            wlnew         = [wlnew;wls];
            mlnew         = [mlnew;mls];
            Plnew         = [Plnew;Pls];
            mEKF          = [mEKF;mls];
            PEKF          = [PEKF;Pls];
        end
        % set the weights, means, and covariances at time tl equal to the
        % set of weights, means, and covariances which include the splits
        wl = wlnew;
        ml = mlnew;
        Pl = Plnew;
    else
        % if the events function did not trigger a stop, everything is done
        % and tl = tk, so break out of the while loop
        break
    end
    
    % compute the new number of components to be used on the next cycle
    nC   = length(wl);
    % cycle the time, weights, means, and covariances
    tlm1 = tl;
    wlm1 = wl;
    mlm1 = ml;
    Plm1 = Pl;
    disp(['Splitting... t = ', num2str(tl), ', nC = ', num2str(nC)])
end

% set the weights, means, and covariances at time tk equal to those at time
% tl
wk = wl;
mk = ml;
Pk = Pl;

end