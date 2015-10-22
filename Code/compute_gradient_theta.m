function grad = compute_gradient_theta(prob, x, theta)
% Compute the gradient for theta

switch prob.loss_flag
    case 'mix3'
        principalIndex = prob.principalIndex;
        generalIndex = prob.generalIndex;
        principalValues = prob.principalValues;
        hinge_rating = prob.hinge_rating;
        edgeThresholdIndex = prob.edgeThresholdIndex;
        rating_levels = prob.rating_levels;
        
        grad = zeros(size(theta));
        for r = 1:length(theta)
            T = ones(size(x.on_omega));
            T(prob.data > rating_levels(r)) = -1;
            tmp0 = (theta(r) - x.on_omega);
            tmp1 = tmp0 .* T;
            tmp2 = 1 - tmp1;
            tmp3 = max(0, tmp2);
            tmp4 = tmp3 .* -T;
%             grad(r) = sum(tmp4);
            grad(r) = sum(tmp4(principalIndex));
        end
    case 'mix2'
        principalIndex = prob.principalIndex;
        generalIndex = prob.generalIndex;
        principalValues = prob.principalValues;
        hinge_rating = prob.hinge_rating;
        edgeThresholdIndex = prob.edgeThresholdIndex;
        rating_levels = prob.rating_levels;
        
        grad = zeros(size(theta));
        for r = 1:length(theta)
            if ~any(r==edgeThresholdIndex)
                continue
            end
            T = ones(size(x.on_omega));
            T(prob.data > rating_levels(r)) = -1;
            tmp0 = (theta(r) - x.on_omega);
            tmp1 = tmp0 .* T;
            tmp2 = 1 - tmp1;
            tmp3 = max(0, tmp2);
            tmp4 = tmp3 .* -T;
%             grad(r) = sum(tmp4);
            grad(r) = sum(tmp4(principalIndex));
        end
    case 'mix'
        principalIndex = prob.principalIndex;
        generalIndex = prob.generalIndex;
        principalValues = prob.principalValues;
        hinge_rating = prob.hinge_rating;
        
        grad = zeros(size(theta));
        for r = 1:length(theta)
            T = ones(size(x.on_omega));
            T(prob.data > hinge_rating(r)) = -1;
            tmp0 = (theta(r) - x.on_omega);
            tmp1 = tmp0 .* T;
            tmp2 = 1 - tmp1;
            tmp3 = max(0, tmp2);
            tmp4 = tmp3 .* -T;
%             grad(r) = sum(tmp4);
            grad(r) = sum(tmp4(principalIndex));
        end
    case 'mix1'
        grad = zeros(size(theta));
        for r = 1:length(theta)
            if any(r==prob.edgeThresholdIndex)
                T = ones(size(x.on_omega));
                T(prob.data > prob.rating_levels(r)) = -1;
                tmp0 = (theta(r) - x.on_omega);
                tmp1 = tmp0 .* T;
                tmp2 = 1 - tmp1;
                tmp3 = max(0, tmp2);
                tmp4 = tmp3 .* -T;
                grad(r) = sum(tmp4);
            end
        end
    case 'SMM'
        mapping = prob.mapping;
        mappedData = mapping(prob.data);
        valueTrunk = prob.valueTrunk;
        
        grad = zeros(size(theta));
        for r = 1:length(theta)
            T = ones(size(x.on_omega));
            T(mappedData > valueTrunk(r)) = -1;
            tmp0 = (theta(r) - x.on_omega);
            tmp1 = tmp0 .* T;
            tmp2 = 1 - tmp1;
            tmp3 = max(0, tmp2);
            tmp4 = tmp3 .* -T;
            grad(r) = sum(tmp4);
        end
                
    case 'SMM2'
        principalIndex = prob.principalIndex;
        generalIndex = prob.generalIndex;
        rating_levels = prob.rating_levels;
        edgeThresholdIndex = prob.edgeThresholdIndex;
        
        principalData = prob.data(principalIndex);
        generalData = prob.data(generalIndex);
        principalX = x.on_omega(principalIndex);
        generalX = x.on_omega(generalIndex);
        
        grad = zeros(size(theta));
        for r = 1:length(theta)
%             if any(r==edgeThresholdIndex)
                % principal values
                T = ones(size(x.on_omega));
                T(prob.data > rating_levels(r)) = -1;
                tmp0 = (theta(r) - x.on_omega);
                tmp1 = tmp0 .* T;
                tmp2 = 1 - tmp1;
                tmp3 = max(0, tmp2);
                tmp4 = tmp3 .* -T;
                grad(r) = sum(tmp4);
%             else
%                 % general values
%                 grad(r) = grad(r) + sum(theta(r) - 1 - generalX);
%             end
        end
    case 'SMM3'
        theta_bk = prob.theta_bk;
        principalIndex = prob.principalIndex;
        generalIndex = prob.generalIndex;
        rating_levels = prob.rating_levels;
        edgeThresholdIndex = prob.edgeThresholdIndex;
        
        principalData = prob.data(principalIndex);
        generalData = prob.data(generalIndex);
        principalX = x.on_omega(principalIndex);
        generalX = x.on_omega(generalIndex);
        
        grad = zeros(size(theta));
        for r = 1:length(theta)
            % principal values
            T = ones(size(principalX));
            T(principalData > rating_levels(r)) = -1;
            tmp0 = (theta(r) - principalX);
            tmp1 = tmp0 .* T;
            tmp2 = 1 - tmp1;
            tmp3 = max(0, tmp2);
            tmp4 = tmp3 .* -T;
            grad(r) = sum(tmp4);
            
%             % general values
%             Q = zeros(size(generalX));
%             Q(rating_levels(r)==generalData) = 0.5;
%             Q(rating_levels(r)==generalData-1) = 0.5;
%             grad(r) = grad(r) + sum(Q.*(Q*theta(r)-generalX));
        end
        
        % general values
        theta_bk2 = theta_bk; theta_bk2(2:end-1) = theta;
        grad_general = zeros(size(theta_bk2));
        for z = 1:length(theta_bk2)
            Q = zeros(size(generalX));
            Q(z==generalData) = 0.5;
            Q(z==generalData-1) = 0.5;
            grad_general(z) = grad_general(z) + sum(Q .* (Q*theta_bk2(z) - generalX));
        end
        grad = grad + grad_general(2:end-1);
        
    case 'SMM4'
        theta_bk = prob.theta_bk;
        principalIndex = prob.principalIndex;
        generalIndex = prob.generalIndex;
        rating_levels = prob.rating_levels;
        edgeThresholdIndex = prob.edgeThresholdIndex;
        
        principalData = prob.data(principalIndex);
        generalData = prob.data(generalIndex);
        principalX = x.on_omega(principalIndex);
        generalX = x.on_omega(generalIndex);
        
        grad = zeros(size(theta));
        theta_bk2 = theta_bk; theta_bk2(2:end-1) = theta;
        for r = 1:length(theta)
            % principal values
            T = ones(size(principalX));
            T(principalData > rating_levels(r)) = -1;
            tmp0 = (theta(r) - principalX);
            tmp1 = tmp0 .* T;
            tmp2 = 1 - tmp1;
            tmp3 = max(0, tmp2);
            tmp4 = tmp3 .* -T;
            grad(r) = sum(tmp4);
            
            % general values
            z = r;
            grad(r) = grad(r) + sum(0.5*(0.5*(theta_bk2(generalData(generalData==z)+1)+theta_bk2(generalData(generalData==z))))); % note the index of theta_bk2
            grad(r) = grad(r) + sum(0.5*(0.5*(theta_bk2(generalData(generalData==1+z)+1)+theta_bk2(generalData(generalData==1+z)))));
        end

    case 'SMM5'
        theta_bk = prob.theta_bk;
        principalIndex = prob.principalIndex;
        generalIndex = prob.generalIndex;
        rating_levels = prob.rating_levels;
        edgeThresholdIndex = prob.edgeThresholdIndex;
        
        principalData = prob.data(principalIndex);
        generalData = prob.data(generalIndex);
        principalX = x.on_omega(principalIndex);
        generalX = x.on_omega(generalIndex);
        
        grad = zeros(size(theta));
        theta_bk2 = theta_bk; theta_bk2(2:end-1) = theta;
        for r = 1:length(theta)
            % principal values
            if ~isempty(principalIndex)
                T = ones(size(principalX));
                T(principalData > rating_levels(r)) = -1;
                tmp0 = (theta(r) - principalX);
                tmp1 = tmp0 .* T;
                tmp2 = 1 - tmp1;
                tmp3 = max(0, tmp2);
                tmp4 = tmp3 .* -T;
                grad(r) = sum(tmp4);
            end
            
            z = r - 1;
            % general values
            if ~isempty(generalIndex)
                grad(r) = grad(r) + sum(theta_bk2(z+1) - generalData(generalData==z+1));
                grad(r) = grad(r) + sum(theta_bk2(z+1) - generalData(generalData==z));
            end
        end
        
%         % general values
%         theta_bk2 = theta_bk; theta_bk2(2:end-1) = theta;
%         grad_general = zeros(size(theta_bk2));
%         for z = 1:length(theta_bk2)
%             Q = zeros(size(generalX));
%             Q(z==generalData) = 0.5;
%             Q(z==generalData-1) = 0.5;
%             grad_general(z) = grad_general(z) + sum(Q .* (Q*theta_bk2(z) - generalX));
%         end
%         grad = grad + grad_general(2:end-1);





%     % below is useful and effective
%     case 'SMM2'
%         grad = zeros(size(theta));
%         for r = 1:length(theta)
%             T = ones(size(x.on_omega));
%             T(prob.data > prob.rating_levels(r)) = -1;
%             tmp0 = (theta(r) - x.on_omega);
%             tmp1 = tmp0 .* T;
%             tmp2 = 1 - tmp1;
%             tmp3 = max(0, tmp2);
%             tmp4 = tmp3 .* -T;
%             grad(r) = sum(tmp4);
%         end
%     % above is useful and effective
        
    case 'RMM'
        grad = zeros(size(theta));
        for r = 1:length(theta)
           T = ones(size(x.on_omega));
           T(prob.data > r) = -1;
        %    sum(T(T==-1))
        %    grad(r) = sum(max(0, 1 - (theta(r) - x.on_omega) .* T) .* -T);
           tmp0 = (theta(r) - x.on_omega);
           tmp1 = tmp0 .* T;
           tmp2 = 1 - tmp1;
           tmp3 = max(0, tmp2);
           tmp4 = tmp3 .* -T;
           grad(r) = sum(tmp4);
        end
    case 'RMM-revised'
        grad = zeros(size(theta));
        for r = 1:length(theta)
            T = ones(size(x.on_omega));
            T(prob.data > prob.rating_levels(r)) = -1;
            tmp0 = (theta(r) - x.on_omega);
            tmp1 = tmp0 .* T;
            tmp2 = 1 - tmp1;
            tmp3 = max(0, tmp2);
            tmp4 = tmp3 .* -T;
            grad(r) = sum(tmp4);
        end
end
