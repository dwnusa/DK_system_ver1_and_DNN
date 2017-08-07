%% 가상 데이터 시뮬레이션
clear all;
count = 3000;
c1count = count;
c1mean  = [5 5];
c1var   = 1.5;
c1 = [c1mean(1)+c1var*randn(c1count,1) c1mean(2)+c1var*randn(c1count,1)];

c2count = count;
c2mean  = [10 10];
c2var   = 1.5;
c2 = [c2mean(1)+c2var*randn(c2count,1) c2mean(2)+c2var*randn(c2count,1)];

c3count = count;
c3mean  = [15 15];
c3var   = 1.5;
c3 = [c3mean(1)+c3var*randn(c3count,1) c3mean(2)+c3var*randn(c3count,1)];

c4count = count;
c4mean  = [20 20];
c4var   = 1.5;
c4 = [c4mean(1)+c4var*randn(c4count,1) c4mean(2)+c4var*randn(c4count,1)];

figure,
subplot(2,2,1), 
plot( c1(:,1), c1(:,2), 'ro' ); hold on;
plot( c2(:,1), c2(:,2), 'bo' ); hold on;
plot( c3(:,1), c3(:,2), 'go' ); hold on;
plot( c4(:,1), c4(:,2), 'co' ); hold off;
title('scatter plot'); xlabel('A'); ylabel('B');
subplot(2,2,3), 
histogram(c1(:,1),50); hold on; 
histogram(c2(:,1),50); hold on;
histogram(c3(:,1),50); hold on;
histogram(c4(:,1),50); hold off;
title('x axis histogram'); xlabel('A'); ylabel('count');
subplot(2,2,4), 
histogram(c1(:,2),50); hold on; 
histogram(c2(:,2),50); hold on;
histogram(c3(:,2),50); hold on;
histogram(c4(:,2),50); hold off;
title('y axis histogram'); xlabel('B'); ylabel('count');
X = [c1; c2; c3; c4];
L = [repmat(1,size(c1,1),1); repmat(2,size(c2,1),1); repmat(3,size(c3,1),1); repmat(4,size(c4,1),1)];
[Y, W, lambda] = LDA(X, L);
Y = Y*50;
y1 = Y(1:count,:);
y2 = Y(count+1:2*count,:);
y3 = Y(2*count+1:3*count,:);
y4 = Y(3*count+1:4*count,:);
subplot(2,2,2), 
histogram(y1(:,1),50); hold on; 
histogram(y2(:,1),50); hold on;
histogram(y3(:,1),50); hold on;
histogram(y4(:,1),50); hold off;
title('B=A axis histogram'); xlabel('B=A'); ylabel('count');

ctrain = [  c1' ...
            c2' ...
            c3' ...
            c4' ...
         ];
ctarget = [ repmat([10], 1, count) ...
            repmat([20], 1, count) ...
            repmat([30], 1, count) ...
            repmat([40], 1, count) ...
          ];

size_Trainset = size(ctrain,2);
shuffle_idx = randperm(size_Trainset);
shuffled_Trainset = ctrain(:,shuffle_idx);
shuffled_Target = ctarget(:,shuffle_idx);
%% 개별 픽셀 포인트소스 데이터 합치기 (옵션)
cnt = 100000;
tempSample = zeros(4,cnt*143);
index = 0;
for row_idx = 1:12
    for column_idx = 1:12
        pix_idx = [row_idx column_idx];
        Label = (pix_idx(1)-1)*12+pix_idx(2); % 1부터 시작하도록 120 뺌
        file_column_idx = column_idx;
        filename = ['time0_C' num2str(file_column_idx) 'R' num2str(row_idx) '.mat']
        try
            load(filename);
        catch
           continue; 
        end
        Sample = Sample(:,1:cnt);
        tempSample(:,(1:size(Sample,2))+size(Sample,2)*index) = Sample;
        index = index + 1;
    end
end
Sample = tempSample;
save('Sample_time4.mat', 'Sample');
%% 학습용 데이터 추출
clear all;close all;clc;
load('Sample_time4.mat');
FEATURE = Sample(:,1:size(Sample,2));
% ANGER 1
xysum = double([(FEATURE(1,:) + FEATURE(2,:)); (FEATURE(1,:) + FEATURE(3,:)); (sum(FEATURE(1:4,:)));]);
figure(1), 
% subplot(2,2,1), histogram(xysum(3,:),[0:20:3000]);
x = min(500, max(1, round(499*(xysum(1,:)./(xysum(3,:))))+1));
y = min(500, max(1, round(499*(xysum(2,:)./(xysum(3,:))))+1));
xy = [x; y];
sp = sparse(y, x, 1, 500, 500);
im = full(sp);
subplot(1,2,1), imagesc(im'); set(gca, 'YDir', 'normal'); title('im');title('Uniform source image');
subplot(1,2,2), mesh(sp'); view(2); title('im 3D');
%% som example
temp_xy = xy;
size_xy = size(temp_xy,2);
shuffle_idx = randperm(size_xy);
shuffled_xy = xy(:,shuffle_idx);

% X = [c1; c3]';
X = shuffled_xy(:,1:10000);
% figure(1), plot(X(1,:),X(2,:),'+r');
close all;
dimensions   = [24 24];
coverSteps   = 200;
initNeighbor = 1;
topologyFcn  = 'gridtop';
distanceFcn  = 'linkdist';
net = selforgmap(dimensions,coverSteps,initNeighbor,topologyFcn,distanceFcn);
net = configure(net,X);
% figure(1), plotsompos(net,X)
net.trainParam.epochs = 100;
net = train(net,X);% ,'useGPU','yes');
% figure(2), plotsompos(net)
figure(3), plotsompos(net,X)
%%
x = 1:500;
y = 1:500;
[X,Y] = meshgrid(x, y);
figure,
for i = 500
    for j = 500
        z = net([X(j) Y(i)]);
        % pause(0.1);
    end
end
plot(z);
%% k means clustering
total = (sum(FEATURE));
temp = double(FEATURE./repmat(total,4,1)).*100;
temp_xy = xy;
size_xy = size(temp,2);
shuffle_idx = randperm(size_xy);
shuffled_temp = temp(:,shuffle_idx);
shuffled_xy = xy(:,shuffle_idx);
range = 1:size_xy/10;
idx = kmeans(shuffled_temp(:,range)', 144);
tempim = zeros(500,500);
figure,
for i = 1:144
   sp = sparse(shuffled_xy(2,idx==i),shuffled_xy(1,idx==i),1,500,500);
   im = full(sp);
   tempim = tempim + im;
   % imagesc(tempim');
   subplot(1,2,1), plot(shuffled_xy(1,idx==i), shuffled_xy(2,idx==i), '*', 'MarkerSize',2); hold on;
   mean_x = mean(shuffled_xy(1,idx==i));
   mean_y = mean(shuffled_xy(2,idx==i));
   subplot(1,2,2), plot(mean_x, mean_y, 'o', 'MarkerSize', 4); hold on;
   pause(0.1);
end
figure, imagesc(tempim');
%%
% xx = 1:1:500;
% yy = 1:1:500;
% [xxG, yyG] = meshgrid(xx,yy);
% xyGrid = [xxG(:), yyG(:)];
% 
% idx2Region = kmeans(xyGrid, 144, 'MaxIter', 1,'Start',C);
% figure;
% gscatter(xyGrid(:,1),xyGrid(:,2), idx2Region);
% hold on;
% plot(xy(1,range), xy(2,range), 'k*','MarkerSize',1);
%% 영상처리 픽셀분할 (워터쉐드 적용)
figure(2),
img1 = im;
se1 = strel('disk',5);
masksize1 = 5;
temp_img1 = imgaussfilt(img1, 8);
img1 = max(0,img1 - temp_img1);
img1 = imopen(img1,se1);                % figure(5),subplot(2,2,1), imagesc(img1');
img1 = imgaussfilt(img1, 1);
img1 = imgaussfilt(img1, masksize1);    % figure(5),subplot(2,2,2), imagesc(img1');
img1 = imregionalmax(img1);
peak_img1 = img1;
img1 = bwdist(img1);
img1 = watershed(img1);
peak_img1 = double(img1) .* double(peak_img1);
label_img1 = img1;
subplot(2,2,1), imagesc(peak_img1'); title('peak img1');
subplot(2,2,2), imagesc(label_img1'); title('label img1');
subplot(2,2,3), imagesc(img1'); title('img1 (label)');
img1 = im .* (img1 ~= 0) + (img1 == 0) .* max(im(:));
subplot(2,2,4), imagesc(img1'); title('img1 (fusion)');
%% 라벨링 순서 정렬
relabel_img = zeros(500, 500);
figure(3), 
subplot(1,2,1), imagesc(label_img1'); title('label img1');
[x, y, v] = find(peak_img1);
temp_rows = sortrows([y x v]);
temp_rows = reshape(temp_rows(:,3), [12, 12]);
temp_cols = sortrows([x y v]);
temp_cols = reshape(temp_cols(:,3), [12, 12]);
relabel = 0;
for i = 1:12
    rows = temp_rows(:,i);
    for j = 1:12
        relabel = relabel + 1;
        cols = temp_cols(:,j);
        temp0 = ismember(rows, cols);
        [temp1 temp1 temp_label] = find(temp0 .* rows);
        temp2 = (label_img1 == temp_label);
        relabel_img = relabel_img + temp2 .* relabel;
    end
end
subplot(1,2,2), imagesc(relabel_img'); title('relabel img');
%% 픽셀별로 학습데이터 획득
temp2 = zeros(500, 500);
figure(4),
for train_label = 1:144
    temp = (relabel_img == train_label);
    temp = img1 .* temp;
    threshold = max(temp(:))*0.2;
    temp1 = max(0, temp-threshold);
    temp2 = temp2 + temp1;
end
mask = logical(temp2);
training_label_mask = mask .* relabel_img;
subplot(2,2,1), imagesc(temp2'); title('temp2');
subplot(2,2,2), imagesc(training_label_mask'); title('training label mask');

for train_label = 1:144
    temp = (relabel_img == train_label);
    temp = img1 .* temp;
    threshold = max(temp(:))*0.3;
    temp1 = max(0, temp-threshold);
    temp2 = temp2 + temp1;
end
mask = logical(temp2);
test_label_mask = mask .* relabel_img;
subplot(2,2,3), imagesc(temp2'); title('temp2');
subplot(2,2,4), imagesc(test_label_mask'); title('test label mask');

train_label = zeros(1,size(xy,2));
test_label = zeros(1,size(xy,2));
for i = 1:size(xy,2)
    train_label(i) = training_label_mask(xy(2,i), xy(1,i));
    test_label(i) = test_label_mask(xy(2,i), xy(1,i));
end

% train_idx = logical(train_label);
% test_idx = logical(test_label);
% train_xy = xy(:,train_idx);
% train_label = train_label(train_idx);

%% 학습 데이터 숫자 일치시키기
figure(4),
train_cnt = zeros(12,12);
trainset_idx = [];
testset_idx = [];
count = 2000;
for i = 1:144
    train_idx = find(train_label == i);
    % train_cnt(i) = size(train_label(train_idx),2);
    train_idx = train_idx(1:count);
    %test_idx = train_idx(count+1:end);
    trainset_idx = [trainset_idx train_idx];
    test_idx = find(test_label == i);
    % train_cnt(i) = size(test_label(test_idx),2);
    test_idx = test_idx(1:count);
    testset_idx = [testset_idx test_idx];
end

% subplot(2,2,3), imagesc(train_cnt'); title('training data count');
% sp = sparse(train_cnt);
% subplot(2,2,4), mesh(sp);

%% 학습용 데이터 영상으로 확인
temp = xy(:,trainset_idx);
sp = sparse(temp(2,:), temp(1,:), 1, 500, 500);
im = full(sp);
figure(5), 
subplot(1,2,1), imagesc(im'); title('trainset');
temp = xy(:,testset_idx);
sp = sparse(temp(2,:), temp(1,:), 1, 500, 500);
im = full(sp);
figure(5), 
subplot(1,2,2), imagesc(im'); title('testset');
%% 학습데이터 평가
figure(31),
xlim = 50;
ylim = 50;
temp_trainset = [];
temp_target = [];
temp_label = [];
for i = 61:72
    idx = mod(i-1,12)+1;
    Temp_ = FEATURE(:,Label==i);
    subplot(2,3,1), scatter(FEATURE(1,:), FEATURE(2,:), 2); hold on; title('A-B'); xlabel('A'); ylabel('B'); axis([0 xlim 0 ylim]);
    subplot(2,3,2), scatter(Temp_(1,:), Temp_(3,:), 2); hold on; title('A-C'); xlabel('A'); ylabel('C'); axis([0 xlim 0 ylim]);
    subplot(2,3,3), scatter(Temp_(1,:), Temp_(4,:), 2); hold on; title('A-D'); xlabel('A'); ylabel('D'); axis([0 xlim 0 ylim]);
    subplot(2,3,4), scatter(Temp_(2,:), Temp_(3,:), 2); hold on; title('B-C'); xlabel('B'); ylabel('C'); axis([0 xlim 0 ylim]);
    subplot(2,3,5), scatter(Temp_(2,:), Temp_(4,:), 2); hold on; title('B-D'); xlabel('B'); ylabel('D'); axis([0 xlim 0 ylim]);
    subplot(2,3,6), scatter(Temp_(3,:), Temp_(4,:), 2); hold on; title('C-D'); xlabel('C'); ylabel('D'); axis([0 xlim 0 ylim]);
    % subplot(2,4,7), scatter(Temp_(1,:)+Temp_(2,:), Temp_(1,:)+Temp_(3,:), 2); hold on; title('AB-AC'); xlabel('AB'); ylabel('AC');
    temp_trainset = [temp_trainset Temp_];
    temp_target = [temp_target repmat(30+idx*50, 1, size(Temp_, 2))];
    temp_label = [temp_label repmat(idx,1,size(Temp_,2))];
end

%% 학습용 데이터 준비
Trainset_ABCD = FEATURE(:,trainset_idx);
Testset_ABCD = FEATURE(:,testset_idx);
Trainset = [Trainset_ABCD(1,:); Trainset_ABCD(2,:); Trainset_ABCD(3,:); Trainset_ABCD(4,:)] ./ repmat(sum(Trainset_ABCD),4,1) * 100;
Tesetset = [Testset_ABCD(1,:); Testset_ABCD(2,:); Testset_ABCD(3,:); Testset_ABCD(4,:)] ./ repmat(sum(Testset_ABCD),4,1) * 100;
% temp = Trainset ./ 1000;
% xx = round((temp(1,:)+temp(2,:))*499)+1;
% yy = round((temp(1,:)+temp(3,:))*499)+1;
% sp = sparse(yy, xx, 1, 500, 500);
% im = full(sp);
% figure(5), imagesc(im');
[X Y] = meshgrid(1:12,1:12);
X = repmat(X(:),1,count); X = reshape(X',1,size(X(:),1));
Y = repmat(Y(:),1,count); Y = reshape(Y',1,size(Y(:),1));
Target = [X; Y] .* 30 + 55;
save('param.mat');
%% 신경망 학습
clear all; close all; clc;
load('param.mat');
%%
size_Trainset = size(Trainset,2);
shuffle_idx = randperm(size_Trainset);
shuffled_Trainset = Trainset(:,shuffle_idx);
shuffled_Target = Target(:,shuffle_idx);
clc;
% reset(gpuDevice(1));
netXY = fitnet([800 600 400 200 20 10]);% hiddenLayerSize/2]); 100 50 40 30 20 10              800 600 400 300 200 20 10
% netXY = fitnet([1000 800 600 300 100 20]);% hiddenLayerSize/2]); 100 50 40 30 20 10
% netXY_2.layers{1}.transferFcn = 'poslin';
% netXY_2.layers{2}.transferFcn = 'poslin';
netXY.layers{3}.transferFcn = 'poslin';
% netXY_2.layers{4}.transferFcn = 'poslin';
% netXY_2.layers{5}.transferFcn = 'poslin';
% netXY_2.layers{6}.transferFcn = 'poslin';
% netXY_2.layers{7}.transferFcn = 'poslin';
netXY.divideParam.trainRatio = 70/100;
netXY.divideParam.valRatio = 15/100;
netXY.divideParam.testRatio = 15/100;
% Train the Network
numC = size(shuffled_Trainset(1,:),2);
numC = round(numC * 0.2);
% Train the Network
[netXY tr] = train(netXY, (shuffled_Trainset(:,1:numC)),shuffled_Target(:,1:numC),'useGPU','yes'); % 'useParallel','yes',
save('netXY.mat','netXY');
%% 학습검증
load('netXY.mat');
numC = size(shuffled_Trainset(1,:),2);
numC = round(numC * 0.3);

% Energy = sum(Verifset(5,numC+1:end));
Verifset = shuffled_Trainset(1:4,numC+1:end)./100;%.*repmat(Energy,4,1);
r_Verif_Pos = round(499.*[(Verifset(1,:)+Verifset(2,:)); (Verifset(1,:)+Verifset(3,:))])+1;
Verif_img = sparse(r_Verif_Pos(2,:), r_Verif_Pos(1,:), 1,500,500);
img11 = (full(Verif_img));
figure(5), subplot(2,2,1), imagesc(img11');

% Trainset
out = netXY((shuffled_Trainset(1:4,numC+1:end)));
r_out = max(1, min(500, round(out)));
out_img = sparse(r_out(2,:), r_out(1,:), 1, 500, 500);
img12 = full(out_img);% rot90(full(sp')',2);
figure(5), subplot(2,2,2), imagesc(img12'); 
%% 실제검증
% load('Sample_time1.mat');%('time1_point_10cm_x2_LOW_thr_100.mat');

load('netXY.mat');
clearvars -except netXY Sample label_img1 peak_img1;clc;

Sample1 = Sample(:,1:size(Sample,2)/10);
Energy = sum(Sample1);
Verifset = Sample1./repmat(Energy,4,1);%.*repmat(Energy,4,1);
before_XY = round(499.*[(Verifset(1,:)+Verifset(2,:)); (Verifset(1,:)+Verifset(3,:))])+1;
Verif_img = sparse(before_XY(2,:), before_XY(1,:), 1,500,500);
img21 = (full(Verif_img));
figure(5), subplot(2,2,3), imagesc(img21');

% Trainset
Testset = [Verifset * 100];
after_XY = [];
after_XY = [after_XY netXY( Testset( : , 1                         :   size(Testset,2)   ))];
% after_XY = [after_XY netXY( Testset( : , size(Testset,2)*1/10+1    :   size(Testset,2)*2/10     ))];
% after_XY = [after_XY netXY( Testset( : , size(Testset,2)*2/10+1    :   size(Testset,2)*3/10     ))];
% after_XY = [after_XY netXY( Testset( : , size(Testset,2)*3/10+1    :   size(Testset,2)*4/10     ))];
% after_XY = [after_XY netXY( Testset( : , size(Testset,2)*4/10+1    :   size(Testset,2)*5/10     ))];
% after_XY = [after_XY netXY( Testset( : , size(Testset,2)*5/10+1    :   size(Testset,2)*6/10     ))];
% after_XY = [after_XY netXY( Testset( : , size(Testset,2)*6/10+1    :   size(Testset,2)*7/10     ))];
% after_XY = [after_XY netXY( Testset( : , size(Testset,2)*7/10+1    :   size(Testset,2)*8/10     ))];
% after_XY = [after_XY netXY( Testset( : , size(Testset,2)*8/10+1    :   size(Testset,2)*9/10     ))];
% after_XY = [after_XY netXY( Testset( : , size(Testset,2)*9/10+1    :   size(Testset,2)*10/10     ))];
after_XY = max(1, min(500, round(after_XY)));
out_img = sparse(after_XY(2,:), after_XY(1,:), 1, 500, 500);
img22 = full(out_img);
figure(5), subplot(2,2,4), imagesc(img22');
%% scatter hist
IM = img21;
IM2 = imcomplement(IM);
IM2 = rot90(IM2,2);
figure, imagesc(IM2');
figure, scatterhist(before_XY(1,:), before_XY(2,:),'Kernel','on','MarkerSize',1,'LineWidth',1);%'MarkerSize',1);
%%
IM = img22;
IM2 = imcomplement(IM);
IM2 = rot90(IM2,2);
figure, imagesc(IM2');
figure, scatterhist(after_XY(1,:), after_XY(2,:),'Kernel','on','MarkerSize',1,'LineWidth',1);
%%
figure(1),
img1 = img21;
se1 = strel('disk',2);
temp_img1 = imgaussfilt(img1, 15);
img1 = max(0,img1 - temp_img1);
img1 = imopen(img1,se1);                % figure(5),subplot(2,2,1), imagesc(img1');
img1 = imgaussfilt(img1, 3);
img1 = imgaussfilt(img1, 3);    % figure(5),subplot(2,2,2), imagesc(img1');
img1 = imregionalmax(img1);
peak_img1 = img1;
img1 = bwdist(img1);
img1 = watershed(img1);
peak_img1 = double(img1) .* double(peak_img1);
label_img1 = img1;
subplot(2,2,1), imagesc(peak_img1'); title('peak img1');
subplot(2,2,2), imagesc(label_img1'); title('label img1');
subplot(2,2,3), imagesc(img1'); title('img1 (label)');
img1 = img21 .* (img1 ~= 0) + (img1 == 0) .* max(img21(:));
subplot(2,2,4), imagesc(img1'); title('img1 (fusion)');
%%
figure(2),
img1 = img22;
se1 = strel('disk',2);
temp_img1 = imgaussfilt(img1, 30);
img1 = max(0,img1 - temp_img1);
img1 = imopen(img1,se1);                % figure(5),subplot(2,2,1), imagesc(img1');
img1 = imgaussfilt(img1, 3);
img1 = imgaussfilt(img1, 3);    % figure(5),subplot(2,2,2), imagesc(img1');
img1 = imregionalmax(img1);
peak_img1 = img1;
img1 = bwdist(img1);
img1 = watershed(img1);
peak_img1 = double(img1) .* double(peak_img1);
label_img1 = img1;
subplot(2,2,1), imagesc(peak_img1'); title('peak img1');
subplot(2,2,2), imagesc(label_img1'); title('label img1');
subplot(2,2,3), imagesc(img1'); title('img1 (label)');
img1 = img22 .* (img1 ~= 0) + (img1 == 0) .* max(img22(:));
subplot(2,2,4), imagesc(img1'); title('img1 (fusion)');
%%
IM = img1;
IM2 = imcomplement(IM);
IM2 = rot90(IM2,2);
figure, imagesc(IM2');
%% 선형성 개선 확인
% peak to valley ratio
X1 = sum(img21');
X2 = sum(img22');
Y1 = sum(img21);
Y2 = sum(img22);
figure(5), 
subplot(4,1,1), plot(1:500,X1); title('Y축');
subplot(4,1,2), plot(1:500,X2); title('Y축');
subplot(4,1,3), plot(1:500,Y1); title('X축');
subplot(4,1,4), plot(1:500,Y2); title('X축');

rangeX1 = 98:400;
maxX1 = max(X1(rangeX1));
minX1 = maxX1 - max(maxX1 - X1(rangeX1));
p2vX1 = maxX1/minX1;
rangeX2 = 83:417;
maxX2 = max(X2(rangeX2));
minX2 = maxX2 - max(maxX2 - X2(rangeX2));
p2vX2 = maxX2/minX2;
rangeY1 = 82:428;
maxY1 = max(Y1(rangeY1));
minY1 = maxY1 - max(maxY1 - Y1(rangeY1));
p2vY1 = maxY1/minY1;
rangeY2 = 83:417;
maxY2 = max(Y2(rangeY2));
minY2 = maxY2 - max(maxY2 - Y2(rangeY2));
p2vY2 = maxY2/minY2;
disp([num2str(p2vX1) ', ' num2str(p2vX2) ', ' num2str(p2vY1) ', ' num2str(p2vY2)]);

figure(6), 
subplot(4,1,1), plot(1:500,X1, [min(rangeX1(:)) min(rangeX1(:))], [0 max(X1(:))], [max(rangeX1(:)) max(rangeX1(:))], [0 max(X1(:))]); title('Y축');
subplot(4,1,2), plot(1:500,X2, [min(rangeX2(:)) min(rangeX2(:))], [0 max(X2(:))], [max(rangeX2(:)) max(rangeX2(:))], [0 max(X2(:))]); title('Y축');
subplot(4,1,3), plot(1:500,Y1, [min(rangeY1(:)) min(rangeY1(:))], [0 max(Y1(:))], [max(rangeY1(:)) max(rangeY1(:))], [0 max(Y1(:))]); title('X축');
subplot(4,1,4), plot(1:500,Y2, [min(rangeY2(:)) min(rangeY2(:))], [0 max(Y2(:))], [max(rangeY2(:)) max(rangeY2(:))], [0 max(Y2(:))]); title('X축');
%% 영상처리 픽셀분할 (워터쉐드 적용)
img1 = img21;
img2 = img22;
masksize1 = 5;
masksize2 = 10;

se1 = strel('disk',5);
se2 = strel('disk',1);
%
% temp_img1 = imgaussfilt(img1, 8);
% img1 = max(0,img1 - temp_img1);
% img1 = imopen(img1,se1);                % figure(5),subplot(2,2,1), imagesc(img1');
% img1 = imgaussfilt(img1, 1);
% img1 = imgaussfilt(img1, masksize1);    % figure(5),subplot(2,2,2), imagesc(img1');
% img1 = imregionalmax(img1);
% peak_img1 = img1;
% img1 = bwdist(img1);
% img1 = relabel_img1;
% %;
% peak_img1 = double(img1) .* double(peak_img1);
% label_img1 = relabel_img;
% peak_img1

temp_img2 = imgaussfilt(img2, 200);
img2 = max(0,img2 - temp_img2);
               %figure(5),subplot(2,2,3), imagesc(img2');
% img2 = imopen(img2,se2);
% img2 = imopen(img2,se2);
for i = 1:1
    img2 = imopen(img2,se2); 
end
for i = 1:20
    blurimg2 = imgaussfilt(img2, 20);
    img2 = max(0,img2 - blurimg2);
end
img2 = imgaussfilt(img2, 6);
% img2 = max(0,img2 - blurimg2);
% img2 = imgaussfilt(img2, 2);
% img2 = imgaussfilt(img2, masksize2);    %figure(5),subplot(2,2,4), imagesc(img2');
img2 = imregionalmax(img2);
peak_img2 = img2;
img2 = bwdist(img2);
img2 = watershed(img2);
peak_img2 = double(img2) .* double(peak_img2);
label_img2 = img2;

% figure(7), 
% subplot(2,2,1), imagesc(img21'); 
% subplot(2,2,2), imagesc(img22');
% subplot(2,2,3), imagesc(img1'); 
% subplot(2,2,4), imagesc(img2');
% figure(8), 
% subplot(1,2,1), imagesc(img1'); 
% subplot(1,2,2), imagesc(img2');


line_temp1 = img21 .* (img1 ~= 0) + (img1 == 0) .* max(img21(:));
line_temp2 = img22 .* (img2 ~= 0) + (img2 == 0) .* max(img22(:));

figure(9), 
subplot(1,2,1), imagesc(line_temp1'); 
subplot(1,2,2), imagesc(line_temp2');

%% 라벨링 순서 정렬
relabel_img1 = zeros(500, 500);
[x, y, v] = find(peak_img1);
temp_rows = sortrows([y x v]);
temp_rows = reshape(temp_rows(:,3), [12, 12]);
temp_cols = sortrows([x y v]);
temp_cols = reshape(temp_cols(:,3), [12, 12]);
relabel = 0;
for i = 1:12
    rows = temp_rows(:,i);
    for j = 1:12
        relabel = relabel + 1;
        cols = temp_cols(:,j);
        temp0 = ismember(rows, cols);
        [temp1 temp1 temp_label] = find(temp0 .* rows);
        temp2 = (label_img1 == temp_label);
        relabel_img1 = relabel_img1 + temp2 .* relabel;
    end
end
figure(10), subplot(1,2,1), imagesc(relabel_img1'); title('relabel img');

relabel_img2 = zeros(500, 500);
[x, y, v] = find(peak_img2);
temp_rows = sortrows([y x v]);
temp_rows = reshape(temp_rows(:,3), [12, 12]);
temp_cols = sortrows([x y v]);
temp_cols = reshape(temp_cols(:,3), [12, 12]);
relabel = 0;
for i = 1:12
    rows = temp_rows(:,i);
    for j = 1:12
        relabel = relabel + 1;
        cols = temp_cols(:,j);
        temp0 = ismember(rows, cols);
        [temp1 temp1 temp_label] = find(temp0 .* rows);
        temp2 = (label_img2 == temp_label);
        relabel_img2 = relabel_img2 + temp2 .* relabel;
    end
end
figure(10), subplot(1,2,2), imagesc(relabel_img2'); title('relabel img');
%% 픽셀별 에너지 스펙트럼 비교 
before_XY_label = zeros(1,size(before_XY,2));
after_XY_label = zeros(1,size(before_XY,2));
for i = 1:size(before_XY,2)
    before_XY_label(i) = relabel_img1(before_XY(2,i), before_XY(1,i));
    after_XY_label(i) = relabel_img2(after_XY(2,i), after_XY(1,i));
end

spXY1 = zeros(500, 500);
spXY2 = zeros(500, 500);

cntmat1 = zeros(12,12);
cntmat2 = zeros(12,12);
k = 0;

for i = 1:144
    before_idx = (before_XY_label == i);
    tempXY1 = before_XY(:,before_idx);
    spXY1 = spXY1 + sparse(tempXY1(2,:), tempXY1(1,:), 1,500,500);
    Verif_img1 = full(spXY1);
    figure(1), subplot(2,2,1), imagesc(Verif_img1');
    
    after_idx = (after_XY_label == i);
    tempXY2 = after_XY(:,after_idx);
    spXY2 = spXY2 + sparse(tempXY2(2,:), tempXY2(1,:), 1,500,500);
    Verif_img2 = full(spXY2);
    figure(1), subplot(2,2,2), imagesc(Verif_img2);
    
    figure(1), subplot(2,2,3), histogram(Energy(:,before_idx), [0:50:14000]); axis([0 14000 0 300]); hold on;
    figure(1), subplot(2,2,3), histogram(Energy(:,after_idx), [0:50:14000]); axis([0 14000 0 300]); hold off;
    
    cntmat1(i) = sum(before_idx(:));
    cntmat2(i) = sum(after_idx(:));
    %pause(0.1);
    w = waitforbuttonpress;
    if w == 0
        disp('Button click')
    else
        disp('Key press')
    end
end
%% 균등성 확인 & 카운트 비교
figure, 
subplot(1,2,1), imagesc(cntmat1);
subplot(1,2,2), imagesc(cntmat2);
mean1 = mean(cntmat1(:)); mean1 = repmat(mean1,12,12);
mean2 = mean(cntmat2(:)); mean2 = repmat(mean2,12,12);

mse1 = immse(cntmat1, mean1)
mse2 = immse(cntmat2, mean2)
sum(cntmat1(:))
sum(cntmat2(:))
%% 픽셀 분할 에러 검증용 데이터 획득
IM = zeros(500, 500);
IM2 = zeros(500, 500);
IM3 = zeros(500, 500);
IM4 = zeros(500, 500);
testset_data = [];
testset_label = [];
test_label = {};
for row_idx = 1:4
    for column_idx = 1:8
        pix_idx = [row_idx column_idx];
        Label = (pix_idx(1)-1)*12+pix_idx(2); % 1부터 시작하도록 120 뺌
        file_column_idx = column_idx;
        filename = ['time0_C' num2str(file_column_idx) 'R' num2str(row_idx) '.mat']
        try
            load(filename);
        catch
           continue; 
        end
        
        SampleE = sum(Sample);
        SampleXY = round([(Sample(1,:)+Sample(2,:))./SampleE*499; (Sample(1,:)+Sample(3,:))./SampleE*499])+1;
        SampleSP = sparse(SampleXY(2,:), SampleXY(1,:), 1, 500, 500);
        SampleIM = full(SampleSP);
        threshold = max(SampleIM(:))*0.4;
        Mask = max(0, SampleIM-threshold); 
        Mask = imgaussfilt(Mask, 5);
        IMmax = max(Mask(:));
        Mask = Mask ./ IMmax;
        IM = IM + Mask;
        
        Mask = max(0, Mask-0.005);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        IM2 = IM2 + Mask;
        IM3 = IM3 + Mask .* SampleIM;
        
        Mask = logical(Mask).*Label;
        IM4 = IM4 + Mask;
        test_label{Label} = zeros(1,size(Sample,2));
        for i = 1:size(Sample,2)
            test_label{Label}(i) = Mask(SampleXY(2,i), SampleXY(1,i));
        end
        temp = Sample(:,test_label{Label}~=0);
        testset_data = [testset_data temp];
        testset_label = [testset_label repmat(Label,1,size(temp,2))];
    end
end
line = (label_img1 ~= 0);
line1 = (label_img1 == 0) .* max(IM(:));
IM = IM .* line + line1;

figure,
subplot(2,2,1), imagesc(IM');
subplot(2,2,2), imagesc(IM2');
% IM3 = max(0, IM-0.05);
subplot(2,2,3), imagesc(logical(IM3)');
subplot(2,2,4), imagesc(IM4');

% index = (testset_label == 7);
testset_data1 = testset_data(:,:);
figure(12), 
testset_dataE = sum(testset_data1);
testset_dataXY = round([(testset_data1(1,:)+testset_data1(2,:))./testset_dataE*499; (testset_data1(1,:)+testset_data1(3,:))./testset_dataE*499])+1;
testset_dataSP = sparse(testset_dataXY(2,:), testset_dataXY(1,:), 1, 500, 500);
testset_dataIM = full(testset_dataSP);
subplot(1,2,1), imagesc(testset_dataIM');
testset_dataABCD = testset_data1./repmat(testset_dataE,4,1).*100;
out = netXY(testset_dataABCD);
out = max(1, min(500, round(out)));
out_img = sparse(out(2,:), out(1,:), 1, 500, 500);
out_img = full(out_img);
subplot(1,2,2), imagesc(out_img');
%% 픽셀분할
line1 = (relabel_img1==0);
line2 = (relabel_img2==0);
figure,
subplot(2,2,1), imagesc(line1');
subplot(2,2,2), imagesc(line2');
temp1 = testset_dataIM.*(~line1) +line1.*max(testset_dataIM(:));
temp2 = out_img .* (~line2) +line2.*max(out_img(:));
subplot(2,2,3), imagesc(temp1');
subplot(2,2,4), imagesc(temp2');
%
% i = 141;
% figure,
% subplot(1,2,1), imagesc((relabel_img1 == i)');
% subplot(1,2,2), imagesc((relabel_img2 == i)');
%% True Positive 추출
PT1 = 0;
PT2 = 0;
list = unique(testset_label);
E1 = zeros(1, size(list,2));
E2 = zeros(1, size(list,2));
% E12 = zeros(2, size(list,2));
TP1 = zeros(1, size(list,2));
TP2 = zeros(1, size(list,2));
TotalSample = zeros(1, size(list,2));
relabelimg1 = 145 - relabel_img1;
relabelimg2 = 145 - relabel_img2;
% i = 2;
% figure,
% subplot(1,2,1), imagesc((relabelimg1 == i)');
% subplot(1,2,2), imagesc((relabelimg2 == i)');
k = 0;
for i = list
    k = k + 1;
    index = (testset_label == i);
    original_setXY = testset_dataXY(:,index);
    classified_setXY = out(:,index);
    classified_label1 = zeros(1,size(original_setXY,2));
    classified_label2 = zeros(1,size(original_setXY,2));
    for j = 1:size(original_setXY,2)
        classified_label1(j) = relabelimg1(original_setXY(2,j), original_setXY(1,j)) == i;
%         relabel_img1(original_setXY(2,j), original_setXY(1,j));
%         relabel_img2(out(2,j), out(1,j))
        classified_label2(j) = relabelimg2(classified_setXY(2,j), classified_setXY(1,j)) == i;
    end
    TP1(k) = sum(classified_label1(:)); E1(k) = (size(classified_label1,2) - TP1(k))/size(classified_label1,2);
    TP2(k) = sum(classified_label2(:)); E2(k) = (size(classified_label2,2) - TP2(k))/size(classified_label2,2);
    TotalSample(k) = size(classified_label1,2);
    disp(['PT1: ' num2str(sum(classified_label1(:))) '/' num2str(size(classified_label1,2)) ]);
    disp(['PT2: ' num2str(sum(classified_label2(:))) '/' num2str(size(classified_label2,2)) ]);
end

E1 = reshape(E1,8,4); % TP1 = reshape(TP1,8,4);
E2 = reshape(E2,8,4); % TP2 = reshape(TP2,8,4);
figure, 
subplot(4,1,1), 
m1 = plot(E1(1:8,1), 'o-'); M1 = 'Anger'; hold on; 
m2 = plot(E2(1:8,1), 'x-'); M2 = 'DNN'; 
legend([m1, m2], M1, M2);
title('Anger, DNN Error along row1');
hold off; 
subplot(4,1,2), 
m1 = plot(E1(1:8,2), 'o-'); M1 = 'Anger'; hold on; 
m2 = plot(E2(1:8,2), 'x-'); M2 = 'DNN'; 
legend([m1, m2], M1, M2);
title('Anger, DNN Error along row2');
hold off; 
subplot(4,1,3), 
m1 = plot(E1(1:8,3), 'o-'); M1 = 'Anger'; hold on; 
m2 = plot(E2(1:8,3), 'x-'); M2 = 'DNN'; 
legend([m1, m2], M1, M2);
title('Anger, DNN Error along row3');
hold off; 
subplot(4,1,4), 
m1 = plot(E1(1:8,4), 'o-'); M1 = 'Anger'; hold on; 
m2 = plot(E2(1:8,4), 'x-'); M2 = 'DNN'; 
legend([m1, m2], M1, M2);
title('Anger, DNN along Error along row4');
hold off; 
sum(TP1(:))
sum(TP2(:))
sum(TotalSample(:))
try
    load('TP12.mat');
catch
    TP12 = [];
end
TP12 = [TP12; [TP1(:) TP2(:) TotalSample(:)]];
save('TP12.mat', 'TP12');

%%
clear all; close all;clc;
load('TP12_performance_test1.mat');
e12 = [TP12(:,3)-TP12(:,1) TP12(:,3)-TP12(:,2)];
e1 = reshape(e12(:,1),32,6);
e2 = reshape(e12(:,2),32,6);
total = reshape(TP12(:,3),32,6);
mean_e1 = mean(e1');
mean_e2 = mean(e2');
mean_total = mean(total');
re_mean_e1 = reshape(mean_e1, 8, 4);
re_mean_e2 = reshape(mean_e2, 8, 4);
re_mean_total = reshape(mean_total,8,4);
error_rate1 = re_mean_e1./re_mean_total;
error_rate2 = re_mean_e2./re_mean_total;
figure(1), 
subplot(4,1,1), plot(re_mean_e1(:,1),'o-'); hold on; plot(re_mean_e2(:,1),'x-'); title('row1'); xlabel('pixel ID'); ylabel('pixel cnt'); row1 = mean(re_mean_e1(:,1) - re_mean_e2(:,1));
subplot(4,1,2), plot(re_mean_e1(:,2),'o-'); hold on; plot(re_mean_e2(:,2),'x-'); title('row2'); xlabel('pixel ID'); ylabel('pixel cnt'); row2 = mean(re_mean_e1(:,2) - re_mean_e2(:,2));
subplot(4,1,3), plot(re_mean_e1(:,3),'o-'); hold on; plot(re_mean_e2(:,3),'x-'); title('row3'); xlabel('pixel ID'); ylabel('pixel cnt'); row3 = mean(re_mean_e1(:,3) - re_mean_e2(:,3));
subplot(4,1,4), plot(re_mean_e1(:,4),'o-'); hold on; plot(re_mean_e2(:,4),'x-'); title('row4'); xlabel('pixel ID'); ylabel('pixel cnt'); row4 = mean(re_mean_e1(:,4) - re_mean_e2(:,4));
figure(2),
subplot(4,1,1), plot(error_rate1(:,1),'o-'); hold on; plot(error_rate2(:,1),'x-'); title('row1'); xlabel('pixel ID'); ylabel('error rate'); rate_row1 = row1/mean(re_mean_e1(:,1));
subplot(4,1,2), plot(error_rate1(:,2),'o-'); hold on; plot(error_rate2(:,2),'x-'); title('row2'); xlabel('pixel ID'); ylabel('error rate'); rate_row2 = row2/mean(re_mean_e1(:,2));
subplot(4,1,3), plot(error_rate1(:,3),'o-'); hold on; plot(error_rate2(:,3),'x-'); title('row3'); xlabel('pixel ID'); ylabel('error rate'); rate_row3 = row3/mean(re_mean_e1(:,3));
subplot(4,1,4), plot(error_rate1(:,4),'o-'); hold on; plot(error_rate2(:,4),'x-'); title('row4'); xlabel('pixel ID'); ylabel('error rate'); rate_row4 = row4/mean(re_mean_e1(:,4));
Tot = mean([row1 row2 row3 row4]);
rate_Tot = Tot / mean(re_mean_e1(:))

%%
clear all; close all;clc;
load('TP12_performance_test2.mat');
e12 = [TP12(:,3)-TP12(:,1) TP12(:,3)-TP12(:,2)];
e1 = reshape(e12(:,1),32,2); 
e2 = reshape(e12(:,2),32,2);
total = reshape(TP12(:,3),32,2);
mean_e1 = mean(e1');
mean_e2 = mean(e2');
mean_total = mean(total');
re_mean_e1 = reshape(mean_e1, 8, 4);
re_mean_e2 = reshape(mean_e2, 8, 4);
re_mean_total = reshape(mean_total,8,4);
error_rate1 = re_mean_e1./re_mean_total;
error_rate2 = re_mean_e2./re_mean_total;
% figure(3), 
% subplot(4,1,1), plot(re_mean_e1(:,1),'o-'); hold on; plot(re_mean_e2(:,1),'x-'); title('row1'); xlabel('pixel ID'); ylabel('pixel cnt');
% subplot(4,1,2), plot(re_mean_e1(:,2),'o-'); hold on; plot(re_mean_e2(:,2),'x-'); title('row2'); xlabel('pixel ID'); ylabel('pixel cnt');
% subplot(4,1,3), plot(re_mean_e1(:,3),'o-'); hold on; plot(re_mean_e2(:,3),'x-'); title('row3'); xlabel('pixel ID'); ylabel('pixel cnt');
% subplot(4,1,4), plot(re_mean_e1(:,4),'o-'); hold on; plot(re_mean_e2(:,4),'x-'); title('row4'); xlabel('pixel ID'); ylabel('pixel cnt');
% figure(4),
% subplot(4,1,1), plot(error_rate1(:,1),'o-'); hold on; plot(error_rate2(:,1),'x-'); title('row1'); xlabel('pixel ID'); ylabel('error rate');
% subplot(4,1,2), plot(error_rate1(:,2),'o-'); hold on; plot(error_rate2(:,2),'x-'); title('row2'); xlabel('pixel ID'); ylabel('error rate');
% subplot(4,1,3), plot(error_rate1(:,3),'o-'); hold on; plot(error_rate2(:,3),'x-'); title('row3'); xlabel('pixel ID'); ylabel('error rate');
% subplot(4,1,4), plot(error_rate1(:,4),'o-'); hold on; plot(error_rate2(:,4),'x-'); title('row4'); xlabel('pixel ID'); ylabel('error rate');
figure(1), 
subplot(4,1,1), plot(re_mean_e1(:,1),'o-'); hold on; plot(re_mean_e2(:,1),'x-'); title('row1'); xlabel('pixel ID'); ylabel('pixel cnt'); row1 = mean(re_mean_e1(:,1) - re_mean_e2(:,1));
subplot(4,1,2), plot(re_mean_e1(:,2),'o-'); hold on; plot(re_mean_e2(:,2),'x-'); title('row2'); xlabel('pixel ID'); ylabel('pixel cnt'); row2 = mean(re_mean_e1(:,2) - re_mean_e2(:,2));
subplot(4,1,3), plot(re_mean_e1(:,3),'o-'); hold on; plot(re_mean_e2(:,3),'x-'); title('row3'); xlabel('pixel ID'); ylabel('pixel cnt'); row3 = mean(re_mean_e1(:,3) - re_mean_e2(:,3));
subplot(4,1,4), plot(re_mean_e1(:,4),'o-'); hold on; plot(re_mean_e2(:,4),'x-'); title('row4'); xlabel('pixel ID'); ylabel('pixel cnt'); row4 = mean(re_mean_e1(:,4) - re_mean_e2(:,4));
figure(2),
subplot(4,1,1), plot(error_rate1(:,1),'o-'); hold on; plot(error_rate2(:,1),'x-'); title('row1'); xlabel('pixel ID'); ylabel('error rate'); rate_row1 = row1/mean(re_mean_e1(:,1));
subplot(4,1,2), plot(error_rate1(:,2),'o-'); hold on; plot(error_rate2(:,2),'x-'); title('row2'); xlabel('pixel ID'); ylabel('error rate'); rate_row2 = row2/mean(re_mean_e1(:,2));
subplot(4,1,3), plot(error_rate1(:,3),'o-'); hold on; plot(error_rate2(:,3),'x-'); title('row3'); xlabel('pixel ID'); ylabel('error rate'); rate_row3 = row3/mean(re_mean_e1(:,3));
subplot(4,1,4), plot(error_rate1(:,4),'o-'); hold on; plot(error_rate2(:,4),'x-'); title('row4'); xlabel('pixel ID'); ylabel('error rate'); rate_row4 = row4/mean(re_mean_e1(:,4));
Tot = mean([row1 row2 row3 row4]);
rate_Tot = Tot / mean(re_mean_e1(:))