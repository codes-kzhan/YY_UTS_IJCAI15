function Grad = gradmy(prob,x,theta)
%GRAD   Computes the gradient on the manifold
%
%  Computes the gradient at a point x of the cost function on 
%  the manifold. + 0.01*x.U +0.01*x.V

switch prob.loss_flag
        
    case 'LS'
        d = x.on_omega - prob.data;
    case 'RMM'
        if nargin < 3
            theta = prob.theta;
        end
        d = zeros(size(x.on_omega));
        for r=1:length(theta)
           T = ones(size(x.on_omega));
           T(prob.data>r) = -1;
           err = T .* max(0, 1 - T .* (theta(r) - x.on_omega));
           d = d + err;
        end
    case 'RMM-revised'
        if nargin < 3
            theta = prob.theta;
        end
        d = zeros(size(x.on_omega));
        for r=1:length(theta)
           T = ones(size(x.on_omega));
           T(prob.data>prob.rating_levels(r)) = -1;
           err = T .* max(0, 1 - T .* (theta(r) - x.on_omega));
           d = d + err;
        end
    case '1MM'
        err = max(0, 1 - x.on_omega .* prob.data);
        d = -prob.data .* err;
end

% d is the gradient in the Euclidean space
% below is the code for the gradient on the manifold

updateSval(prob.temp_omega, d, length(d));
Grad = prob.temp_omega;

% T = prob.temp_omega*x.V;
% g.M = x.U'*T; 
% g.Up = T - x.U*(x.U'*T);
% 
% g.Vp = prob.temp_omega'*x.U; 
% g.Vp = g.Vp - x.V*(x.V'*g.Vp);

    
    

