% Returns the azimuth and elevation in radians [2x1] Array
% R = diag([(0.5 * pi / 180)^2, (0.5 * pi / 180)^2]);
% add_noise is either {true, false}
function measurements = angles_only(positions, mu, R, add_noise)
    x = positions(:,1) + mu;
    y = positions(:,2);
    z = positions(:,3);
    
    alpha = atan2(y, x);
    epsilon = asin(z ./ sqrt(x.^2 + y.^2 + z.^2));
    measurements = [alpha epsilon];
    
    if add_noise 
        measurements = mvnrnd(measurements, R);
    end
end