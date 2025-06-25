function [p,X,Y,M,Ptot] = gmmpdf(w,mu,Si,v,G,sigmas)

% this function computes a 2D PDF of a GMM
% inputs:
% w = weights of the components
% mu = mean of the components
% Si = covariance matrix of the components
% v = vectors with the index of the components to plot
% G = number of points in each coordiate of the grid
% sigmas = number of sigmas to plot for each component




c    = length(w);
xmin = +inf;
xmax = -inf;
ymin = +inf;
ymax = -inf;
nx = length(mu{1,1});
V    = cell(c,1);
D    = cell(c,1);
M2 = zeros(nx);
M = zeros(nx,1);
for k = 1:c    
    M2 = M2 + w{k}*(Si{k}+mu{k}*mu{k}');
    M = M + w{k}*mu{k};
end

Ptot = M2 - M*M';

for k = 1:c    
    
    m = mu{k}(v);
    P = Si{k}(v,v);
    
    V{k} = inv(P);
    D{k}  = 1/sqrt(det(2*pi*P));

    xmink = m(1) - sigmas*sqrt(P(1,1));
    xmaxk = m(1) + sigmas*sqrt(P(1,1));
    ymink = m(2) - sigmas*sqrt(P(2,2));
    ymaxk = m(2) + sigmas*sqrt(P(2,2));
    
    if(xmink < xmin), xmin = xmink; end
    if(xmaxk > xmax), xmax = xmaxk; end
    if(ymink < ymin), ymin = ymink; end
    if(ymaxk > ymax), ymax = ymaxk; end
end

xvec  = linspace(xmin,xmax,G);
yvec  = linspace(ymin,ymax,G);
[X,Y] = meshgrid(xvec',yvec');

p = zeros(size(X));
for k = 1:c
    Xk = X-mu{k}(v(1));
    Yk = Y-mu{k}(v(2));
    qf = V{k}(1,1)*Xk.*Xk + V{k}(1,2)*Xk.*Yk + V{k}(2,1)*Xk.*Yk + V{k}(2,2)*Yk.*Yk;
    pe = exp(-0.5*qf);
    p  = p + w{k}*D{k}*pe;
end

end