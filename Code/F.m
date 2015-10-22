function [f,f_principal] = F(prob,x,theta)
%F Cost function 
%

switch prob.loss_flag
        
    case 'LS'
        err = x.on_omega - prob.data; f_principal = 0.5*(err'*err);
        f = f_principal;
    case 'RMM'
        if nargin < 3
            theta = prob.theta;
        end
        f_principal = 0;
        for r=1:size(theta)
           T = ones(size(x.on_omega));
           T(prob.data>r) = -1;
           err = max(0, 1 - T .* (theta(r) - x.on_omega));
           f_principal = f_principal + 0.5 * (err' * err);
        end
        f = f_principal;
    case 'RMM-revised'
        if nargin < 3
            theta = prob.theta;
        end
        f_principal = 0;
        for r=1:size(theta)
           T = ones(size(x.on_omega));
           T(prob.data>prob.rating_levels(r)) = -1;
           err = max(0, 1 - T .* (theta(r) - x.on_omega));
           f_principal = f_principal + 0.5 * (err' * err);
        end
        f = f_principal;
    case '1MM'
        err = max(0, 1 - x.on_omega .* prob.data);
        f = 0.5 * (err' * err);
end


if prob.mu>0
  f = f + 0.5*prob.mu*norm(1./x.sigma)^2 + 0.5*prob.mu*norm(x.sigma)^2;
end