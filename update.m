function [m_update,P_update,U_update] = update(cfig,model,y,R,m,P)
%      
    U_update = zeros(size(m,2),1);
    m_update = zeros(size(m,1),size(m,2));
    P_update = zeros(size(m,1),size(m,1),size(m,2));
    for jj=1:size(m,2)
        % individual EKF update
        ybar = cfig.h(model,m(:,jj),'noiseless');
        Hj   = cfig.H(model,m(:,jj));
        Pxxj = P(:,:,jj);
        Pxyj = Pxxj * Hj';
        Pyyj= Hj*Pxxj*Hj' + R; 
        Pyyj= (Pyyj + Pyyj')/2;   % additional step to avoid numerical problems
        det_Pyyj = prod(eig(Pyyj)); iPyyj = pinv(Pyyj);
        Kj = Pxyj * iPyyj;
        m_update(:,jj)  = m(:,jj) + Kj*(y-ybar);
        Ij = eye(size(m,1));
        P_update(:,:,jj) = (Ij - Kj*Hj) * Pxxj * (Ij - Kj*Hj)' + Kj * R * Kj'; % Joseph form
        
        % weight update
        U_update(jj) = -(y-ybar)' * (iPyyj * (y-ybar)) / 2 ...
                       - log(det_Pyyj) / 2 ...
                       - log(2*pi) * size(y,1) / 2;
    end
%
end