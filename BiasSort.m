function [FrontNo,MaxFNo] = BiasSort(PopObj)
    [N,M] = size(PopObj);
    FrontNo = zeros(1,size(PopObj,1));
    n=2;
    idx_size = M;
    Obj = PopObj;
    t=0;
    while idx_size > 2
       [idx_size, idxs] = divide_into_groups(M, n);
       FrontSet = [];
       for i = 1:length(idxs)
            obj = Obj(:,idxs{i});
            [FN,~] = NDSort(obj,inf);
            FrontSet = [FrontSet;FN];
       end
       % Front = max(FrontSet);
       Front = mean(FrontSet);
       % tag = Front==max(Front);
       tag = Front>0.75*max(Front);
       FrontNo(tag) = N-t;
       Obj = Obj(~tag,:);
       n = n*2;
       t = t+1;
    end
    
    FrontNo = FrontNo-(N-t)+2;
    FrontNo(FrontNo==min(FrontNo))=1;
    MaxFNo = max(FrontNo);
end

function [group_size, groups] = divide_into_groups(N, num_groups)
    % 随机生成 1 到 N 的数据
    data = randperm(N); % 这里用 randperm 打乱顺序

    % 计算每组的大小
    group_size = floor(N / num_groups);
    rem = mod(N, num_groups);

    % 创建分组大小向量
    split_sizes = repmat(group_size, 1, num_groups);
    split_sizes(1:rem) = split_sizes(1:rem) + 1;

    % 分割数据
    groups = mat2cell(data, 1, split_sizes);
end
