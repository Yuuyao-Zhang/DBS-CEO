function Elites = SelectReference(Population, Wb)
    ArcObj = Population.objs;
    
    % DDS
    DDrank = DyDecomposition(ArcObj, Wb) - 1;

    % DDNDS
    [FrontNo, ~] = BiasSort(ArcObj);
    BSrank = FrontNo-1;

    % NDS
    % [FrontNoNDS,~] = NDSort(ArcObj, inf);
    % FNDS = FrontNoNDS - 1; 

    rank = DDrank .* BSrank;
    % rank = BSrank;
    % rank = DDrank;
    % rank = FNDS;
    
    label = rank==0;

    Elites = Population(label);
end