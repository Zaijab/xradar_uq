function Xdot = CR3BP(t, X)
    
    mu = 0.01215;
    
    x = X(1); y = X(2); z = X(3); xdot = X(4); ydot = X(5); zdot = X(6);
    
    ux = 0; uy = 0; uz = 0;
    
    r1 = ((x + mu)^2 + y^2 + z^2)^(1/2);
    r2 = ((x - 1 + mu)^2 + y^2 + z^2)^(1/2);
    
    xddot = 2 * ydot + x -(1-mu)*(x+mu)/r1^3 - mu*(x-1+mu)/r2^3 + ux;
    yddot = -2 * xdot + y -(1-mu)*y/r1^3 - mu*y/r2^3 + uy;
    zddot = -(1-mu)*z/r1^3 - mu*z/r2^3 + uz;
    
    Xdot = [xdot,ydot,zdot,xddot,yddot,zddot]';

end