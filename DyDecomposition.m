function rank = DyDecomposition(obj, V)
    % find extreme solutions
    indexE = [];
    for i = 1 : size(V,1)
         y = TCH(obj, V(i,:));
         [~, indexe] = sort(y, "ascend");
         indexE = [indexE, indexe(1)];
    end
    ExtremeObj = obj(indexE,:);

    Q = ExtremeObj;
    P = obj(setdiff(1:size(obj,1), indexE),:);
    rank = zeros(1, size(obj,1));
    rank(indexE) = 1;

    % calculate DD rank base on Q
    for j = 1 : size(P,1)
        dis = min(pdist2(P, Q),[],2);
        [~, indexp] = sort(dis, 'descend');
        pivot = P(indexp(1),:);
        pivotV = pivot ./ sum(pivot);
        y = TCH(P, pivotV);
        [~, index] = sort(y, 'ascend');
        label = find(all(obj == P(index(1),:), 2));
        rank(label) = j + 1;
        Q = [Q; P(index(1),:)];
        P = P(setdiff(1:size(P,1), index(1)), :);
    end
end

function y = TCH(obj, Vi)
    Zl = min(obj, [], 1); 
    y = max(abs(obj - repmat(Zl, size(obj,1), 1)) .* Vi, [], 2);
end