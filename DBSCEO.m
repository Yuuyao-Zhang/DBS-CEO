classdef DBSCEO < ALGORITHM
% <multi/many> <real/integer> <expensive>
% ne    ---   5 --- The number of solutions to expensive evaluation

%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            ne = Algorithm.ParameterSet(5);

            %% Generate random population
            % NI = 100+Problem.D/10;
            Population    = Problem.Initialization();
            [W, N]  = UniformPoint(Problem.M,Problem.M);
            Z = min(Population.objs, [], 1);

            Wb = eye(Problem.M);
            Wb(Wb==0) = 1e-6;

            %% Optimization
            while Algorithm.NotTerminated(Population)
                evalc('net = newrb(Population.decs'', Population.objs'', 0.01, 1, Problem.D/2,300);');                
                Elites = SelectReference(Population, Wb);
                [OffDec, OffObj] = SolutionGeneration(Problem,Elites,net,Z,W,N);
                candidate = EnvironmentalSelection(Population, OffObj, OffDec, Wb, ne);
                Offspring = Problem.Evaluation(candidate);
                Population = [Population,Offspring];
                Z = min(Population.objs, [], 1);
            end
        end
    end
end