function [value,isterminal,direction] = KLdivstop(t,z,GM,nC,nX,nS,wUKF,Hlm1,dH,mu)
value      = zeros(nC,1);
isterminal = ones(nC,1);   % 1 = the integration terminates when the event occurs
direction  = ones(nC,1);   % 0 = event trigger if value = 0

zj = reshape(z,nX*(1+nS+nX)+1,nC);
wmax = 0;
for j = 1:nC
    if zj(1,j)>wmax
        wmax=zj(1,j);
    end
end
for j = 1:nC
    Xj      = reshape(zj(2:1+nS*nX,j),nX,nS);

    mj_UKF  = zeros(nX,1);
    Pj_UKF  = zeros(nX,nX);
    for iii = 1:nS
        mj_UKF  = mj_UKF  + wUKF(iii)*Xj(:,iii);
        Pj_UKF  = Pj_UKF  + wUKF(iii)*(Xj(:,iii)*Xj(:,iii)');
    end
    Pj_UKF  = Pj_UKF  - mj_UKF*mj_UKF';

    mj_EKF = zj(2+nS*nX:1+(nS+1)*nX,j);
    Pj_EKF = reshape(zj(2+(nS+1)*nX:1+(nX+nS+1)*nX,j),nX,nX);

    d = mj_EKF-mj_UKF;
    KLD_j = 0.5*(log(det(Pj_EKF)/det(Pj_UKF))-nX+...
        d'/Pj_EKF*d+trace(Pj_UKF/Pj_EKF));
    if rcond(Pj_EKF)<1e-15
        a = rcond(Pj_EKF);
    end
    value(j) = zj(1,j)*zj(1,j)*KLD_j - wmax*wmax*dH;
    value(j) = zj(1,j)*KLD_j - wmax*dH;
end

end