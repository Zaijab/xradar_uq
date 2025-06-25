function [dzda] = crtbp_prop(a,z,mu)
% CRTBP dynamics
%   a   - time
%   z   - state vector
%   mu  - cislunar mass ratio

dzda = zeros(size(z));

r = z(1:3);
v = z(4:6);

re = sqrt((r(1) + mu)^2 + r(2)^2 + r(3)^2);
rm = sqrt((r(1) - 1 + mu)^2 + r(2)^2 + r(3)^2);
dzda(1:3) = v;
dzda(4) = r(1) + 2 * v(2) - (1 - mu) * (r(1) + mu) / re^3 - ...
    mu * (r(1) - 1 + mu) / rm^3;
dzda(5) = r(2) - 2 * v(1) - (1 - mu) * r(2) / re^3 - mu * r(2) / rm^3;
dzda(6) = -(1 - mu) * r(3) / re^3 - mu * r(3) / rm^3;
end