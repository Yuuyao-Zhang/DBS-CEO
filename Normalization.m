function ArcObj = Normalization(ArcObj)
    Zl = min(ArcObj, [], 1);
    Zh = max(ArcObj, [], 1);

    diff = abs(Zh-Zl)-(1e-10);
    if any(diff<0)
        Indices = find(diff < 0);
        ReIndices = find(diff > 0);
        ArcObj(:,Indices) = 1e-10;
        ArcObj(:,ReIndices) = (ArcObj(:,ReIndices) - repmat(Zl(ReIndices),size(ArcObj(:,ReIndices),1),1))./(repmat(Zh(ReIndices),size(ArcObj(:,ReIndices),1),1)-repmat(Zl(ReIndices),size(ArcObj(:,ReIndices),1),1));
    else
        ArcObj = (ArcObj - repmat(Zl,size(ArcObj,1),1))./(repmat(Zh,size(ArcObj,1),1)-repmat(Zl,size(ArcObj,1),1));
    end
end