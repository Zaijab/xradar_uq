function H = angles_only_jacobian(state, mu)
    % Jacobian of angles-only measurement function
    x = state(1) + mu;
    y_pos = state(2);
    z = state(3);
    
    rho_norm = sqrt(x^2 + y_pos^2 + z^2);
    rho_xy_norm = sqrt(x^2 + y_pos^2);
    
    % Partial derivatives for azimuth (alpha = atan2(y, x))
    dalpha_dx = -y_pos / (x^2 + y_pos^2);
    dalpha_dy = x / (x^2 + y_pos^2);
    dalpha_dz = 0;
    
    % Partial derivatives for elevation (epsilon = asin(z/rho_norm))
    depsilon_dx = -x * z / (rho_norm^3 * sqrt(1 - (z/rho_norm)^2));
    depsilon_dy = -y_pos * z / (rho_norm^3 * sqrt(1 - (z/rho_norm)^2));
    depsilon_dz = rho_xy_norm^2 / (rho_norm^3 * sqrt(1 - (z/rho_norm)^2));
    
    % Jacobian matrix (2 x 6)
    H = [dalpha_dx,   dalpha_dy,   dalpha_dz,   0, 0, 0;
         depsilon_dx, depsilon_dy, depsilon_dz, 0, 0, 0];
end