clearvars;

datanames = {'musk', 'wbc', 'breastw', 'arrhythmia', 'thyroid'};

for n=1:length(datanames)
    dataname = datanames{n};
    filename = ['./data/raw/', dataname, '.mat'];
    if exist(filename, 'file')
        load(filename);
    else
        continue;
    end

    X_normal = X(y==0,:); num_normal = size(X_normal,1);
    X_anomaly = X(y==1,:); num_anomaly = size(X_anomaly, 1);

    num_train = round((num_normal-num_anomaly)*0.8);

    % shuffle
    rand('state', 12345)
    perm = randperm(num_normal);

    % partition
    X_te = [X_normal(perm(1:num_anomaly),:); X_anomaly]; % always last half is anomaly
    X_tr = X_normal(perm(num_anomaly+1:num_anomaly+num_train),:);
    X_va = X_normal(perm(num_anomaly+num_train+1:end),:);

    % normalize
    m = mean(X_tr,1);
    s = std(X_tr,[],1);
    s(s<1e-6)=1;
    X_tr = bsxfun(@minus, X_tr, m); X_tr = bsxfun(@times, X_tr, 1./s);
    X_va = bsxfun(@minus, X_va, m); X_va = bsxfun(@times, X_va, 1./s);
    X_te = bsxfun(@minus, X_te, m); X_te = bsxfun(@times, X_te, 1./s);

    % save
    mkdir(sprintf('./data/features/%s/', dataname));
    dlmwrite(sprintf('./data/features/%s/data_train.txt',dataname), X_tr, ' ');
    dlmwrite(sprintf('./data/features/%s/data_valid.txt',dataname), X_va, ' ');
    dlmwrite(sprintf('./data/features/%s/data_test.txt',dataname), X_te, ' ');

    fprintf('%s  dim=%3d, tr: %6d, va: %6d, te: %6d\n', dataname, ...
        size(X_tr,2), size(X_tr,1), size(X_va,1), size(X_te,1));
end

