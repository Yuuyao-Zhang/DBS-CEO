function candidate = EnvironmentalSelection(Population, OffObj, OffDec, Wb, ne)
    ArcObj = Population.objs;
    Objs = [ArcObj; OffObj];
    Objs = Normalization(Objs);

    ArcObj = Objs(1 : size(ArcObj,1),:);
    OffObj = Objs(size(ArcObj,1)+1 : end, :);
    [FrontNo, ~] = BiasSort(OffObj);

    label = FrontNo==1;
    OffObj = OffObj(label,:);

    DDrank = DyDecomposition(OffObj, Wb);

    [~, idx] = sort(DDrank, 'ascend');
    
    ne = min(ne, length(idx));
    candidate = OffDec(idx(1:ne),:);

end