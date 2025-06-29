% function [y,discriminator] = GAN3(train_x,M)

function y = GAN(train_x,Problem)

% -----------------定义模型
D = Problem.D;
generator = nnsetup([D, D, D]);
discriminator = nnsetup([D, D, 1]);

batch_size = size(train_x,1);

batch_num = 1;
learning_rate = 0.0001;%学习速率

epoch = 5;


for e=1:epoch
%     kk = randperm(images_num);
    for t=1:batch_num
        % 准备数据
        uniformed_x = train_x;
        noise = unifrnd(0, 1, batch_size, D);%生成batch_size*100阶[0,1]均匀分布数组
        %% 开始训练
        % 第一次先用generator生成一部分数据
        generator = nnff(generator, noise);
        images_fake = generator.layers{generator.layers_count}.a;

        % -----------更新generator，固定discriminator
        % -----------生成器G不断生成“假数据”,判别器D去判断，然后训练G
        discriminator = nnff(discriminator, images_fake);
        logits_fake = discriminator.layers{discriminator.layers_count}.z;
        discriminator = nnbp_d(discriminator, logits_fake, ones(batch_size, 1));
        generator = nnbp_g(generator, discriminator);
        generator = nnapplygrade(generator, learning_rate);

        % -----------不断训练更新discriminator，直到判断是否为假数据的概率为50%
        generator = nnff(generator, noise);
        images_fake = generator.layers{generator.layers_count}.a;
        images = [images_fake;uniformed_x];
        % a = zeros(images_fake,1);
        discriminator = nnff(discriminator, images);
        logits = discriminator.layers{discriminator.layers_count}.z;
        labels = [zeros(size(images_fake,1),1);ones(batch_size,1)];%人为的定义真假样本集的标签，因为希望真样本集的输出尽可能为1，假样本集为0
        discriminator = nnbp_d(discriminator, logits, labels);
        discriminator = nnapplygrade(discriminator, learning_rate);%更新判别器网络的权重
        % ----------------输出loss
        if t == batch_num
            c_loss = sigmoid_cross_entropy(logits(1:batch_size), ones(batch_size, 1));%这是生成器的损失
            d_loss = sigmoid_cross_entropy(logits, labels);%这是判别器的损失
            % fprintf('c_loss:"%f",d_loss:"%f"\n',c_loss, d_loss);
        end
    end
end

%% GAN+DE
% 生成样本
finally_sequence = generator.layers{generator.layers_count}.a;
% 差分进化
P = randperm(size(train_x,1));
DEsolutions = zeros(1,size(train_x,2));
if length(P)>=2
    for i = 1:length(P)
        DE = OperatorDE(Problem, train_x(i), train_x(P(1)), train_x(P(2)));
        if ~ismember(DE,DEsolutions,"rows")
            DEsolutions = [DEsolutions;DE];
        end 
    end
else
    DEsolutions = [];
    % DEsolutions = train_x;
end

candidates = [finally_sequence;DEsolutions];
discriminator = nnff(discriminator, candidates);
count = discriminator.layers{discriminator.layers_count}.a;
[~,rank] = sort(relu(count),'descend');
N = min(size(candidates,1),size(train_x,1));
y = candidates(rank(1:N),:);



%% DE+Generator
% disp('hello');
% 生成样本
% finally_sequence = generator.layers{generator.layers_count}.a;
% P = randperm(size(train_x,1));
% DEsolutions = zeros(1,size(train_x,2));
% if length(P)>=2
%     for i = 1:length(P)
%         DE = OperatorDE(Problem, train_x(i), train_x(P(1)), train_x(P(2)));
%         if ~ismember(DE,DEsolutions,"rows")
%             DEsolutions = [DEsolutions;DE];
%         end 
%     end
% else
%     DEsolutions = [];
%     % DEsolutions = train_x;
% end
% 
% Gsolutions = [finally_sequence;DEsolutions];
% 
% containsNaN = any(~isnan(Gsolutions), 2);
% Gsolutions = Gsolutions(containsNaN,:);
% 
% if isempty(Gsolutions)
%     % 如果没有优秀解，就在负类中选最好的
%     discriminator = nnff(discriminator, images_fake);
% else
%     % 训练辨别器，对每个个体打分
%     discriminator = nnff(discriminator, Gsolutions);
% end
% 
% 
% % count = discriminator.layers{discriminator.layers_count}.a;
% % 得到test中，每个数据的得分值
% count = discriminator.layers{discriminator.layers_count}.a;
% [~,rank] = sort(relu(count),'descend');
% N = min(size(Gsolutions,1),size(train_x,1));
% if isempty(Gsolutions)
%     elite = images_fake(rank(1),:);
%     output = images_fake(rank(1:N),:);
% else
%     elite = Gsolutions(rank(1),:);
%     output = Gsolutions(rank(1:N),:);
% end


% %% DE
% GData = generator.layers{generator.layers_count}.a;
% temp = [train_x;GData];
% P = randperm(size(temp,1));
% DEsolutions = [];
% if length(P)>=2
%     for i = 1:length(P)
%         DE = OperatorDE(Problem, train_x(i), train_x(P(1)), train_x(P(2)));
%         DEsolutions = [DEsolutions;DE];
%     end
% else
%     DEsolutions = GData;
% end
% output = DEsolutions;

% %% G
% % 生成样本
% finally_sequence = generator.layers{generator.layers_count}.a;
% 
% % 将生成的样本与真实样本混合，用于判断discrimination对正类样本的得分值，以此来筛选哪些是正类
% test = [finally_sequence;sequence_real];
% % discriminator = nnff(discriminator, finally_sequence);
% % 训练辨别器，对每个个体打分
% discriminator = nnff(discriminator, test);
% N = size(sequence_real,1); % 正类数据大小
% % 得到test中，每个数据的得分值
% count = discriminator.layers{discriminator.layers_count}.a;
% [low, high] = getMinMax(count(N:end,:));
% 
% % 如果因为正类数据少，所以得到的生成样本少，就再多生成几轮
% 
% % 生成数据的得分值
% generation_count = count(1:N,:);
% % 得到得分值在正类数据得分值的区间索引
% postive_index = find(all(generation_count >= low & generation_count <= high, 2));
% 
% % 如果生成的正类数据太多的话，考虑用一些机制筛选出部分解作为正类解
% % output = generation_count(postive_index);
% c_i = finally_sequence(postive_index,:);
% [~,output] = kmeans(c_i,1);


end


%% Uniform the input
function uniformed_x = UniformInput(x,Problem)
    uniformed_x = ones(1, Problem.D);
    for i = 1 : Problem.D
        x_min = Problem.lower(i);
        x_max = Problem.upper(i);
        uniformed_x(i) = (x(i) - x_min) / (x_max - x_min);
    end
end

function uniformed_x = Uniform(x)
    uniformed_x = zeros(size(x,1),size(x,2));
    expection = zeros(size(x,2),1);
    variance = zeros(size(x,2),1);
    for i = 1 : size(x,1)
        expection(i) = mean(x(i,:));
        variance(i) = var(x(i,:));
        uniformed_x(i,:) = (x(i,:) - expection(i)) / variance(i);
    end
end

% 同时得到正类数据得分的的最大值和最小值
% 目的是选择生成数据得分在这个区间的数据，作为candidate
function [low, high] = getMinMax(x)
    high = max(x);
    low = min(x);
end

% sigmoid激活函数
function output = sigmoid(x)
    output =1./(1+exp(-x));
end
% relu
function output = relu(x)
    output = max(x, 0);
end
% relu对x的导数
function output = delta_relu(x)
    output = max(x,0);
    output(output>0) = 1;
end
% 交叉熵损失函数，此处的logits是未经过sigmoid激活的
% https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
function result = sigmoid_cross_entropy(logits, labels)
    result = max(logits, 0) - logits .* labels + log(1 + exp(-abs(logits)));
    result = mean(result);
end
% sigmoid_cross_entropy对logits的导数，此处的logits是未经过sigmoid激活的
function result = delta_sigmoid_cross_entropy(logits, labels)
    temp1 = max(logits, 0);
    temp1(temp1>0) = 1;
    temp2 = logits;
    temp2(temp2>0) = -1;
    temp2(temp2<0) = 1;
    a = temp1 - labels;
    b = exp(-abs(logits))./(1+exp(-abs(logits)));
    c = b .* temp2;
    result = temp1 - labels + exp(-abs(logits))./(1+exp(-abs(logits))) .* temp2;
end
% 根据所给的结构建立网络
function nn = nnsetup(architecture)
    nn.architecture   = architecture;
    nn.layers_count = numel(nn.architecture);
    % t,beta1,beta2,epsilon,nn.layers{i}.w_m,nn.layers{i}.w_v,nn.layers{i}.b_m,nn.layers{i}.b_v是应用adam算法更新网络所需的变量
    nn.t = 0;
    nn.beta1 = 0.9;
    nn.beta2 = 0.999;
    nn.epsilon = 10^(-8);
    % 假设结构为[100, 512, 784]，则有3层，输入层100，两个隐藏层：100*512，512*784, 输出为最后一层的a值（激活值）
    %
    for i = 2 : nn.layers_count   
        nn.layers{i}.w = normrnd(0, 0.02, nn.architecture(i-1), nn.architecture(i));
        nn.layers{i}.b = normrnd(0, 0.02, 1, nn.architecture(i));
        nn.layers{i}.w_m = 0;
        nn.layers{i}.w_v = 0;
        nn.layers{i}.b_m = 0;
        nn.layers{i}.b_v = 0;
    end
end
% 前向传递
function nn = nnff(nn, x)
    nn.layers{1}.a = x;
    for i = 2 : nn.layers_count
        input = nn.layers{i-1}.a;
        w = nn.layers{i}.w;
        b = nn.layers{i}.b;
        nn.layers{i}.z = input*w + repmat(b, size(input, 1), 1);
        if i ~= nn.layers_count
            % nn.layers{i}.a = relu(nn.layers{i}.z);
            nn.layers{i}.a = tanh(nn.layers{i}.z);
        else
            nn.layers{i}.a = sigmoid(nn.layers{i}.z);
            % nn.layers{i}.z = exp(nn.layers{i}.z);
        end
    end
end
% discriminator的bp，下面的bp涉及到对各个参数的求导
% 如果更改网络结构（激活函数等）则涉及到bp的更改，更改weights，biases的个数则不需要更改bp
% 为了更新w,b，就是要求最终的loss对w，b的偏导数，残差就是在求w，b偏导数的中间计算过程的结果
function nn = nnbp_d(nn, y_h, y)
    % d表示残差，残差就是最终的loss对各层未激活值（z）的偏导，偏导数的计算需要采用链式求导法则-自己手动推出来
    n = nn.layers_count;
    % 最后一层的残差
    nn.layers{n}.d = delta_sigmoid_cross_entropy(y_h, y);
    for i = n-1:-1:2
        d = nn.layers{i+1}.d;
        w = nn.layers{i+1}.w;
        z = nn.layers{i}.z;
        % 每一层的残差是对每一层的未激活值求偏导数，所以是后一层的残差乘上w,再乘上对激活值对未激活值的偏导数
        nn.layers{i}.d = d*w' .* delta_relu(z);    
    end
    % 求出各层的残差之后，就可以根据残差求出最终loss对weights和biases的偏导数
    for i = 2:n
        d = nn.layers{i}.d;
        a = nn.layers{i-1}.a;
        % dw是对每层的weights进行偏导数的求解
        nn.layers{i}.dw = a'*d / size(d, 1);
        nn.layers{i}.db = mean(d, 1);
    end
end
% generator的bp
function g_net = nnbp_g(g_net, d_net)
    n = g_net.layers_count;
    a = g_net.layers{n}.a;
    % generator的loss是由label_fake得到的，(images_fake过discriminator得到label_fake)
    % 对g进行bp的时候，可以将g和d看成是一个整体
    % g最后一层的残差等于d第2层的残差乘上(a .* (a_o))
    g_net.layers{n}.d = d_net.layers{2}.d * d_net.layers{2}.w' .* (a .* (1-a));
    for i = n-1:-1:2
        d = g_net.layers{i+1}.d;
        w = g_net.layers{i+1}.w;
        z = g_net.layers{i}.z;
        % 每一层的残差是对每一层的未激活值求偏导数，所以是后一层的残差乘上w,再乘上对激活值对未激活值的偏导数
        g_net.layers{i}.d = d*w' .* delta_relu(z);    
    end
    % 求出各层的残差之后，就可以根据残差求出最终loss对weights和biases的偏导数
    for i = 2:n
        d = g_net.layers{i}.d;
        a = g_net.layers{i-1}.a;
        % dw是对每层的weights进行偏导数的求解
        g_net.layers{i}.dw = a'*d / size(d, 1);
        g_net.layers{i}.db = mean(d, 1);
    end
end
% 应用梯度
% 使用adam算法更新变量，可以参考：
% https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
function nn = nnapplygrade(nn, learning_rate)
    n = nn.layers_count;
    nn.t = nn.t+1;
    beta1 = nn.beta1;
    beta2 = nn.beta2;
    lr = learning_rate * sqrt(1-nn.beta2^nn.t) / (1-nn.beta1^nn.t);
    for i = 2:n
        dw = nn.layers{i}.dw;
        db = nn.layers{i}.db;
        % 下面的6行代码是使用adam更新weights与biases
        nn.layers{i}.w_m = beta1 * nn.layers{i}.w_m + (1-beta1) * dw;
        nn.layers{i}.w_v = beta2 * nn.layers{i}.w_v + (1-beta2) * (dw.*dw);
        nn.layers{i}.w = nn.layers{i}.w - lr * nn.layers{i}.w_m ./ (sqrt(nn.layers{i}.w_v) + nn.epsilon);
        nn.layers{i}.b_m = beta1 * nn.layers{i}.b_m + (1-beta1) * db;
        nn.layers{i}.b_v = beta2 * nn.layers{i}.b_v + (1-beta2) * (db.*db);
        nn.layers{i}.b = nn.layers{i}.b - lr * nn.layers{i}.b_m ./ (sqrt(nn.layers{i}.b_v) + nn.epsilon); 
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