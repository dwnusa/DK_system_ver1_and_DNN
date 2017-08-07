%% 데이터 생성
close all;
count = 100;
c1count = count;
c1mean  = [17 10];
c1var   = 1;
c11 = [c1mean(1)+c1var*randn(c1count,1) c1mean(2)+c1var*randn(c1count,1)];
c12 = [c1mean(1)+c1var*randn(c1count,1) c1mean(2)+c1var*randn(c1count,1)];
c1 = [c11; c12];
figure(1), scatter(c1(:,1), c1(:,2)); hold on;

c2count = count;
c2mean  = [10 11];
c2var   = 1;
c21 = [c2mean(1)+c2var*randn(c2count,1) c2mean(2)+c2var*randn(c2count,1)];
c22 = [c2mean(1)+c2var*randn(c2count,1) c2mean(2)+c2var*randn(c2count,1)];
c2 = [c21; c22];
figure(1), scatter(c2(:,1), c2(:,2)); hold on;

c3count = count;
c3mean = [10 2];
c3var = 3;
c31 = [c3mean(1)+c3var*randn(c3count,1) c3mean(2)+c3var*randn(c3count,1)];
c32 = [c3mean(1)+c3var*randn(c3count,1) c3mean(2)+c3var*randn(c3count,1)];
c3 = [c31; c32];
figure(1), scatter(c3(:,1), c3(:,2)); hold on;

c4count = count;
c4mean = [17 5];
c4var = 1;
c41 = [c4mean(1)+c4var*randn(c4count,1) c4mean(2)+c4var*randn(c4count,1)];
c42 = [c4mean(1)+c4var*randn(c4count,1) c4mean(2)+c4var*randn(c4count,1)];
c4 = [c41; c42];
figure(1), scatter(c4(:,1), c4(:,2)); hold on;

c5count = count;
c5mean = [12 7];
c5var = 1;
c51 = [c5mean(1)+c5var*randn(c5count,1) c5mean(2)+c5var*randn(c5count,1)];
c52 = [c5mean(1)+c5var*randn(c5count,1) c5mean(2)+c5var*randn(c5count,1)];
c5 = [c51; c52];
figure(1), scatter(c5(:,1), c5(:,2)); hold off;
%% 데이터 목표값 배정
c1 = c1';
c2 = c2';
c3 = c3';
c4 = c4';
c5 = c5';
t1 = repmat(0,1,length(c1));
t2 = repmat(0,1,length(c1));
t3 = repmat(0,1,length(c1));
t4 = repmat(0,1,length(c1));
t5 = repmat(1,1,length(c1));
%% 데이터 병합
C0 = [c1 c2 c3 c4];
C1 = [c5];
T0 = [t1 t2 t3 t4];
T1 = [t5];

%% 데이터 순서섞기
x = [C0 C1];
t = [T0 T1];
shuffle_index = randperm(length(t));
x = x(:,shuffle_index);
t = t(:,shuffle_index);
%% 신경망 학습
% x = x;
% t = t;
% Create a Pattern Recognition Network
hiddenLayerSize = 6;
net = patternnet(hiddenLayerSize);
% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% Train the Network
[net,tr] = train(net,x,t);
% Test the Network
y = net(x);
% View the Network
%view(net)
%% 학습용 그래프랑 인지공간 출력
figure(2), plot(t(1:20),'o'), hold on; plot(y(1:20),'x'); hold off;
[X, Y] = meshgrid(-5:30,-5:20);
index = [X(:) Y(:)]';
out = net(index);
out1 = reshape(out, length(-5:20), length(-5:30));
% out1 = flipud(out1);
figure(2), imagesc(out1);
set(gca,'YDir','normal');
figure(3), mesh(out1);
%% 신경망 피팅 학습

return;
%% 검증용 그래프 출력
a1count = 10;
a1mean  = c1mean;
a1var   = c1var;
a1 = [c1mean(1)+c1var*randn(a1count,1) c1mean(2)+c1var*randn(a1count,1)];

a5count = 10;
a5mean  = c5mean;
a5var   = c5var;
a5 = [c5mean(1)+c5var*randn(a5count,1) c5mean(2)+c5var*randn(a5count,1)];

out_a1 = net(a1');
out_a5 = net(a5');
out_a1 = logical(out_a1(1,:) > out_a5(1,:));
% out_a5 = ~out_a1;

figure(2), plot(out_a1,'o'), hold on; plot(out_a5,'x'); hold off;

