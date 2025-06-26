% Returns the azimuth and elevation in radians [2x1] Array
% add_noise
function measurements = angles_only(positions, add_noise, mu)
    if nargin < 2
        add_noise = false;
    end
    
    x = positions(:,1) + mu;
    y = positions(:,2); 
    z = positions(:,3);
    
    alpha = atan2(y, x);
    epsilon = asin(z ./ sqrt(x.^2 + y.^2 + z.^2));
    measurements = [alpha epsilon];
    
    if add_noise
        covariance = diag([(0.5 * pi / 180)^2, (0.5 * pi / 180)^2]);
        measurements = mvnrnd(measurements', covariance)';
    end
end