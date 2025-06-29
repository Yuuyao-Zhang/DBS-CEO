function [OffDec, OffObj] = SolutionGeneration(Problem,Population,net,Z,W,N)
    maxeva  = 20;
    eva=1;
    nr = 2;
    
    % 为MOEAD做准备
    % [W, N]  = UniformPoint(Problem.N,Problem.M);
    T       = ceil(N/10);
    B       = pdist2(W,W);
    [~,B]   = sort(B,2);
    B       = B(:,1:T);
    gmin    = inf;

    % 进行迭代种群的初始形态
    PopDec  = Population.decs; 
    PopObj = Population.objs;


    while (eva<=maxeva)
        %% MOEA/D
        N1 = min(size(W,1),size(PopDec,1));
        for i = 1 : N1
            if rand < delta
                P = B(i,randperm(size(B,2)));
            else
                P = randperm(N1);
            end
            if length(P)==1
                continue;
            end
            OffDec = OperatorDE(Problem, PopDec(i,:),PopDec(P(1),:),PopDec(P(2),:));
            OffObj = sim(net,OffDec')';
            Z = min(Z,OffObj);
            g_old = max(abs(PopObj(P,:) - repmat(Z,length(P),1)).*W(P,:),[],2);
            g_new = max(repmat(abs(OffObj-Z),length(P),1).*W(P,:),[],2);
            gmin = min([gmin,min(g_old),min(g_new)]);
            offindex = P(find(g_old>g_new,nr));
            if ~isempty(offindex)
                PopDec(offindex,:) = repmat(OffDec,length(offindex),1);
                PopObj(offindex,:) = repmat(OffObj,length(offindex),1);
            end
        end
        Dec1 = PopDec;
        Obj1 = PopObj;


        if mod(eva,5)==0
            %% GAN-generator
            % 用真实的非支配解训练GAN
            GANDec = GAN(Population.decs,Problem);
            GANObj = sim(net,GANDec')';

            Dec2 = GANDec;
            Obj2 = GANObj;

            % select PS in each pop
            [FrontNo1,~] = NDSort(Obj1,inf);
            [FrontNo2,~] = NDSort(Obj2,inf);

            % [FrontNo1,~] = BiasSort(Obj1);
            % [FrontNo2,~] = BiasSort(Obj2);
            tag=1;

            PopDec=[Dec1(FrontNo1==tag,:);Dec2(FrontNo2==tag,:)];
            PopObj=[Obj1(FrontNo1==tag,:);Obj2(FrontNo2==tag,:)];

            % PopDec=[Dec1(FrontNo1==tag,:)];
            % PopObj=[Obj1(FrontNo1==tag,:)];
        end


        % 用IM-MOEA增强
        % [~,partition] = max(1-pdist2(PopObj,W,'cosine'),[],2);
        % for l = unique(partition)'
        %     OffDec = [Operator(Problem,PopDec,PopObj)];
        % end
        % OffObj = sim(net,OffDec')';
        % PopDec = OffDec;
        % PopObj = OffObj;
        
        % delate same solution
        
        eva = eva + 1;

        % OffDec = PopDec;
        % OffObj = PopObj;

        % OffDec = unique(OffDec,"rows");
        % flag = ~ismember(OffDec, Population.decs,"rows");
        % OffDec = OffDec(flag,:);
        % OffObj = OffObj(flag,:);
        % 
        % if isempty(OffObj)
        %     eva = 1;
        % end
    end

   %% Kmeans cluster the solutions into Ke clusters and select the solutions with the maximum EI in each cluster
    % Q = []; temp = Population.decs;
    % for i = 1 : size(PopDec,1)
    %     if min(pdist2(real(PopDec(i,:)),real(temp))) > 1e-5
    %         Q= [Q;PopDec(i,:)];
    %         temp = [temp;PopDec(i,:)];
    %     end
    % end
    % PopDec=Q; 
    % if min(Ke,size(PopDec,1))<=0
    %     PopDec = bestDec;
    % end
    % % cindex  = kmeans(real(PopDec),min(Ke,size(PopDec,1)));
    % cindex  = kmeans(real(PopDec),ceil(min(Ke,size(PopDec,1))));
    % Q = [];
    % for i = 1 : Ke
    %     index = find(cindex == i);
    %     temp = PopDec(index,:);
    %     K = length(index);
    %     tempObj = sim(net,temp')';
    %     if Problem.FE<0.5*Problem.maxFE
    %         EI = EICal(tempObj,1/Problem.M);
    %         [~,best] =min(EI);
    %         Q = [Q,index(best)];
    %     else
    %         dis=zeros(K,1);
    %         for j=1:size(tempObj,1)
    %             [~,dis(j)] =  nbselect(Population.objs,tempObj(j,:),'K',1,Problem.M);
    %         end
    %         [~,best] = max(dis);
    %         Q = [Q,index(best)];
    %     end
    % end
    % PopDec = PopDec(Q,:);

    OffDec = PopDec;
    OffObj = PopObj;

    OffDec = unique(OffDec,"rows");
    flag = ~ismember(OffDec, Population.decs,"rows");
    OffDec = OffDec(flag,:);
    OffObj = OffObj(flag,:);



    if isempty(OffObj)
        [~,partition] = max(1-pdist2(PopObj, W,'cosine'),[],2);
        for l = unique(partition)'
            OffDec = [Operator(Problem,PopDec,PopObj)];
        end
        OffObj = sim(net,OffDec')';
    end
    
end


function OffDec = Operator(Problem,PopDec,PopObj,L)
% The Gaussian process based reproduction

%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is modified from the code in
% http://www.soft-computing.de/jin-pub_year.html

    %% Parameter setting
    if nargin < 4
        L = 3;
    end
    % PopDec = Population.decs;
	% PopObj = Population.objs;
    [N,D]  = size(PopDec);
    
    %% Gaussian process based reproduction
    if size(PopDec,1) < 2*Problem.M
        OffDec = PopDec;
    else
        OffDec = [];
        fmin   = 1.5*min(PopObj,[],1) - 0.5*max(PopObj,[],1);
        fmax   = 1.5*max(PopObj,[],1) - 0.5*min(PopObj,[],1);
        % Train one groups of GP models for each objective
        for m = 1 : Problem.M
            parents = randperm(N,floor(N/Problem.M));
            offDec  = PopDec(parents,:);
            for d = randperm(D,L)
                % Gaussian Process
                try
                    [ymu,ys2] = gp(struct('mean',[],'cov',[],'lik',log(0.01)),...
                                   @infExact,@meanZero,@covLIN,@likGauss,...
                                   PopObj(parents,m),PopDec(parents,d),...
                                   linspace(fmin(m),fmax(m),size(offDec,1))');
                    offDec(:,d) = ymu + rand*sqrt(ys2).*randn(size(ys2));
                catch
                end
            end
            OffDec = [OffDec;offDec];
        end
    end
    
    %% Convert invalid values to random values
    [N,D]   = size(OffDec);
    Lower   = repmat(Problem.lower,N,1);
    Upper   = repmat(Problem.upper,N,1);
    randDec = unifrnd(Lower,Upper);
    invalid = OffDec<Lower | OffDec>Upper;
    OffDec(invalid) = randDec(invalid);

    %% Polynomial mutation
    [proM,disM] = deal(1,20);
    Site   = rand(N,D) < proM/D;
    mu     = rand(N,D);
    temp   = Site & mu<=0.5;
    OffDec = min(max(OffDec,Lower),Upper);
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
end

function dist = EICal(Obj,p)
% Calculate the expected improvement

    M = size(Obj,2);
    d=zeros(size(Obj,1),1);
    for j=1:M
         d=d+abs(Obj(:,j)).^p;
    end
    dist=d.^(1/p);
   
end


function [idx,dist] = nbselect(fitness,part,varargin)
    if varargin{1} == 'K'
        k = varargin{2};
        [idx,dist] = knnsearch(fitness(:,1:end),part(:,1:end),'Distance','euclidean','NSMethod','kdtree','K',k);  
    end
end