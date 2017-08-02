function varargout = Controller(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Controller_OpeningFcn, ...
                   'gui_OutputFcn',  @Controller_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ... 
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:}); 
end


% --- Executes just before Controller is made visible.
function Controller_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
global TEMP;
TEMP = uint32(2690187464); % 0xA05900C8
% --- Outputs from this function are returned to the command line.
function varargout = Controller_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;





% --- Executes on button press in btn_Setup.
function btn_Setup_Callback(hObject, eventdata, handles)
% mex -g Transfer_capture.cpp;
mex -g Transfer_extract.cpp;

% mex -g helloMex.cpp;
% mex -g Transfer_pulse.cpp;
% mex -g Transfer_realtime.cpp;
% mex -g Transfer_reset.cpp;
helloMex(); 




% --- Executes on button press in btn_Start.
function btn_Start_Callback(hObject, eventdata, handles)
% global IM handlesPlot;
close(figure(1));
clear global IM handlesPlot OUT;
clear function;
realVideo();




% --- Executes on button press in btn_Stop.
function btn_Stop_Callback(hObject, eventdata, handles)
close all;
clear function;



% --- Executes on button press in btn_Reset.
function btn_Reset_Callback(hObject, eventdata, handles)
global TEMP;
ep00wire_Callback(hObject, eventdata, handles) ;
ep00wire = TEMP;
Transfer_reset(ep00wire); %out = (Transfer_capture(numRows, numCols, ep00wire));




% --- Executes on button press in btn_Cap.
function btn_Cap_Callback(hObject, eventdata, handles)
global out TEMP Cap_SIZE PULSE_Size;
ep00wire = TEMP;
numRows = int32(4); numCols = int32(Cap_SIZE);
out = (Transfer_capture(numRows, numCols, ep00wire));
if size(out,2) ~= 1
    out = out(:,11:numCols-10);
end
pady = 500; 
minChannel = min(min(out(1:numRows,:)')) - pady; 
maxChannel = max(max(out(1:numRows,:)')) + pady; 
figure(1),subplot(5,1,1), plot(out(1,:)'); axis([0 numCols  minChannel maxChannel]);title('Channel A');ylabel('ADC Value');% xlabel('Samples');
figure(1),subplot(5,1,2), plot(out(2,:)'); axis([0 numCols minChannel maxChannel]);title('Channel B');ylabel('ADC Value');% xlabel('Samples');
figure(1),subplot(5,1,3), plot(out(3,:)'); axis([0 numCols minChannel maxChannel]);title('Channel C');ylabel('ADC Value');% xlabel('Samples');
figure(1),subplot(5,1,4), plot(out(4,:)'); axis([0 numCols minChannel maxChannel]);title('Channel D');ylabel('ADC Value');% xlabel('Samples');
figure(1),subplot(5,1,5), plot(out(1:4,:)');axis([0 numCols minChannel maxChannel]);title('Channel ABCD');ylabel('ADC Value'); xlabel('Samples');







% --- Executes on button press in btn_capture2.
function btn_Capture2_Callback(hObject, eventdata, handles)
global out TEMP Cap_SIZE;
ep00wire = TEMP;
numRows = int32(4); numCols = int32(Cap_SIZE);
out = (Transfer_extract(numRows, numCols, ep00wire));
if size(out,2) ~= 1
    out = out(1:4,11:numCols-10);
end

pady = 500; miny = min(out(:)) - pady; maxy = max(out(:)) + pady;
figure(2),subplot(5,1,1), plot(out(1,:)'); axis([0 numCols  miny maxy]);title('Channel A');ylabel('ADC Value');% xlabel('Samples');
figure(2),subplot(5,1,2), plot(out(2,:)'); axis([0 numCols miny maxy]);title('Channel B');ylabel('ADC Value');% xlabel('Samples');
figure(2),subplot(5,1,3), plot(out(3,:)'); axis([0 numCols miny maxy]);title('Channel C');ylabel('ADC Value');% xlabel('Samples');
figure(2),subplot(5,1,4), plot(out(4,:)'); axis([0 numCols miny maxy]);title('Channel D');ylabel('ADC Value');% xlabel('Samples');
figure(2),subplot(5,1,5), plot(out(1:4,:)');axis([0 numCols miny maxy]);title('Channel ABCD');ylabel('ADC Value'); xlabel('Samples');


% --- Executes on button press in btn_Sample.
function btn_Sample_Callback(hObject, eventdata, handles)
global Stream_SIZE TEMP Cap_SIZE MODE FILENAME FEATURE;
Filename = FILENAME;
h = waitbar(0,'1','Name', 'Sample extract...',...
    'CreateCancelBtn',...
    'setappdata(gcbf,''canceling'',1)');
try
setappdata(h,'canceling',0);
sizx = 512; sizy = 512;
ep00wire = TEMP;
numRows = int32(4); numCols = int32(Cap_SIZE);
Sample = []; XY = [];
IM_Temp = zeros(sizx, sizy); IM_Sample = zeros(sizx, sizy);
cnt = 0;
figure(3), subplot(3,3,1); handlesPlot{1} = imagesc(IM_Temp); colormap(jet);title('Flood Image');
figure(3), subplot(3,3,2); handlesPlot{2} = imagesc(IM_Sample); colormap(jet);title('Flood Image');
figure(3), subplot(3,3,3); handlesPlot{3} = plot(hist(0, 500)); xlim([0 500]); title('Total Energy Spectrum');
% figure(3), subplot(3,3,5); title('Energy Spectrum'); handlesPlot{3} = plot(hist(0, 500)); xlim([0 500]); title('Energy Spectrum');
% figure(3), subplot(3,3,6); title('Energy Spectrum'); handlesPlot{4} = plot(hist(0, 500)); xlim([0 500]); title('Energy Spectrum');
figure(3), subplot(3,3,4); handlesPlot{4} = plot(hist(0, 500)); xlim([0 500]); title('CHA Energy Spectrum');
figure(3), subplot(3,3,5); handlesPlot{5} = plot(hist(0, 500)); xlim([0 500]); title('CHB Energy Spectrum');
figure(3), subplot(3,3,6); handlesPlot{8} = plot(hist(0, 512)); xlim([1 512]); title('X Histogram'); 
figure(3), subplot(3,3,7); handlesPlot{6} = plot(hist(0, 500)); xlim([0 500]); title('CHC Energy Spectrum');
figure(3), subplot(3,3,8); handlesPlot{7} = plot(hist(0, 500)); xlim([0 500]); title('CHD Energy Spectrum');
figure(3), subplot(3,3,9); handlesPlot{9} = plot(hist(0, 512)); xlim([1 512]); title('Y Histogram'); 
    while (cnt < Stream_SIZE)
        if getappdata(h,'canceling')
            break
        end
        Temp = double(Transfer_capture(numRows, numCols, ep00wire));
        if size(Temp,2) ~= 1
            Temp = Temp(:,11:numCols-10);
            if (MODE == false) % DPC
                Energy = sum(Temp); 
                X = min(512, max(1, round((Temp(1,:)+Temp(2,:))./Energy.*300+100)));
                Y = min(512, max(1, round((Temp(1,:)+Temp(3,:))./Energy.*300+100)));
            else  % SCD
                Energy = sum(Temp); 
                X = min(512, max(1, round((Temp(4,:)-Temp(3,:))./Energy.*128*2+256))); %  xx = Math.Round(    image_size / 2 + (image_size / 2) * 3 * (data3[i] - data4[i]) / total  );
                Y = min(512, max(1, round((Temp(2,:)-Temp(1,:))./Energy.*128*2+256))); %  yy = Math.Round(    image_size / 2 + (image_size / 2) * 3 * (data1[i] - data2[i]) / total  );
            end
            % XY = [XY [X; Y]];
            Sample = [Sample Temp];
            cnt = min(Stream_SIZE,size(Sample(1,:),2));
            IM_Temp = rot90(full(sparse(Y,X,1,sizx,sizy))',2);
            IM_Sample = IM_Sample + IM_Temp;
            set(handlesPlot{1},'CData',IM_Temp);% imagesc(IM_Temp); title('Flood Image'); colormap(jet); % Display XYSUM
            set(handlesPlot{2},'CData',IM_Sample);% imagesc(IM_Sample); title('Flood Image'); colormap(jet); % Display XYSUM
            % set(handlesPlot{3},'YData',hist(Energy,500));
            set(handlesPlot{3},'YData',hist(sum(Sample),500));
            set(handlesPlot{4},'YData',hist(Sample(1,:),500));
            set(handlesPlot{5},'YData',hist(Sample(2,:),500));
            set(handlesPlot{6},'YData',hist(Sample(3,:),500));
            set(handlesPlot{7},'YData',hist(Sample(4,:),500));
            set(handlesPlot{8},'YData',hist(X,512));
            set(handlesPlot{9},'YData',hist(Y,512));
        end
        waitbar(cnt/Stream_SIZE,h,sprintf('%d / %d',cnt, Stream_SIZE));
        pause(1/50);
        disp([num2str(size(Temp,2)) ', ' num2str(size(Sample,2))]);
    end
delete(h);
catch
    delete(h);
end
Energy = sum(Sample); 
X = min(512, max(1, round((Sample(1,:)+Sample(2,:))./Energy.*300+100)));
Y = min(512, max(1, round((Sample(1,:)+Sample(3,:))./Energy.*300+100)));
SP = sparse(Y,X,1,sizx,sizy);
% IM_Sample = rot90(full(SP)',2);
% Display X, Y, SUM
figure(4), subplot(1,2,1), imagesc(IM_Sample); title('Flood Image');colormap(jet); % Display XYSUM
figure(4), subplot(1,2,2), mesh(SP);title('3D Flood Image');colormap(jet);
figure(5), subplot(2,2,1), hist(double(Sample(1,:)), 500); title('CHA Energy Spectrum'); % Display XYSUM
figure(5), subplot(2,2,2), hist(double(Sample(2,:)), 500); title('CHB Energy Spectrum'); % Display XYSUM
figure(5), subplot(2,2,3), hist(double(Sample(3,:)), 500); title('CHC Energy Spectrum'); % Display XYSUM
figure(5), subplot(2,2,4), hist(double(Sample(4,:)), 500); title('CHD Energy Spectrum'); % Display XYSUM


% Sample
filename = sprintf('%s',Filename);
if cnt/Stream_SIZE == 1 % (cnt <= Stream_SIZE)
    i = 1;
    while true
        if exist([filename '.mat'],'file') == 2
            filename = sprintf('%s (%d)',Filename,i);
            i = i + 1;
        else
            save([filename '.mat'],'Sample');
            disp('File saved!!');
            msgbox({'File saved!!' [filename '.mat']});
            break;
        end;
    end;
else
    disp('Canceled, Not saved!!');
    msgbox({'Canceled, Not saved!!' [filename '.mat']});
end;


% --- Executes on button press in btn_Extract.
function btn_Extract_Callback(hObject, eventdata, handles)
global FILENAME;
FILENAME = get(handles.txt_filename,'String');
realExtract();


% --- Executes on button press in ep00wire31.
function ep00wire_Callback(hObject, eventdata, handles) 
global TEMP Cap_SIZE Stream_SIZE FILENAME PULSE_Size MODE;
temp = uint32(0);
debug_temp = dec2bin(temp);
if (get(handles.ep00wire31,'Value')), temp = bitset(temp, 32); debug_temp = dec2bin(temp); end;
if (get(handles.ep00wire30,'Value')), temp = bitset(temp, 31); debug_temp = dec2bin(temp); end;
if (get(handles.ep00wire29,'Value')), temp = bitset(temp, 30); debug_temp = dec2bin(temp); end;
if (get(handles.ep00wire28,'Value')), temp = bitset(temp, 29); debug_temp = dec2bin(temp); end;
if (get(handles.ep00wire27,'Value')), temp = bitset(temp, 28); debug_temp = dec2bin(temp); end;
if (get(handles.ep00wire26,'Value')), temp = bitset(temp, 27); debug_temp = dec2bin(temp); end;
if (get(handles.ep00wire25,'Value')), temp = bitset(temp, 26); debug_temp = dec2bin(temp); end;
if (get(handles.ep00wire24,'Value')), MODE = true; else MODE = false; end; % temp = bitset(temp, 25); debug_temp = dec2bin(temp); end;
if (get(handles.ep00wire23,'Value')), end; % temp = bitset(temp, 24); debug_temp = dec2bin(temp); end;
if (get(handles.ep00wire22,'Value')), temp = bitset(temp, 23); debug_temp = dec2bin(temp); end;
if (get(handles.txt_PSt,'Value')), numPS = max(1,min(255,str2double(get(handles.txt_PSt,'String')))); end;
if (get(handles.txt_Thr,'Value')), numThr= max(-8192,min(8191,str2double(get(handles.txt_Thr,'String')))); end;
% if (get(handles.txt_LLD,'Value')), TotalCNT= max(16384,min(134217728,str2double(get(handles.txt_LLD,'String')))); end;
FILENAME = get(handles.txt_filename,'String');
if exist([FILENAME '.mat'],'file') == 2
    msgbox({'File exists already!!' [FILENAME '.mat']});
end
A = uint32(temp);
B = uint32(numPS); shiftB = bitshift(B, 14); set(handles.txt_PSt,'String',num2str(numPS));
C = int32(numThr); C = bitand(int32(16383), C); shiftC = uint32(bitshift(C,0)); set(handles.txt_Thr,'String',num2str(numThr));
D = bitor(A,shiftB);
E = bitor(D,shiftC);
TEMP = E;
debug_temp = dec2bin(TEMP,32);
disp(num2str(['debug_temp : ' num2str(debug_temp) '(32) (' num2str(dec2bin(bitshift(A,-22)),10) ')(10) (' num2str(dec2bin(B,8)) ')(8) (' num2str(dec2bin(C,14)) ')(14)']));
Cap_SIZE = max(1024,min(16384,str2double(get(handles.txt_Csize,'String'))));
Stream_SIZE = max(1024,str2double(get(handles.txt_TotalCnt,'String')));
PULSE_Size = numPS;
disp(['Capture Size : ' num2str(Cap_SIZE)]); 
disp(['Total Count : ' num2str(Stream_SIZE)]);
disp(['MODE : ' num2str(MODE)]);




function realExtract()
global TEMP ELAPSE_TIME FEATURE CNT TotalCNT FILENAME;
ELAPSE_TIME = [];
FEATURE = [];
CNT = 0;
% Define frame rate  
ExtractFeaturesPerSecond=50;

% set up timer object
TimerData=timer('TimerFcn', {@ExtractFeatures,TEMP},'Period',1/ExtractFeaturesPerSecond,'ExecutionMode','fixedRate','BusyMode','drop');

start(TimerData); 
 % Open figure
h = waitbar(0,'1','Name', 'feature extracting...',...
    'CreateCancelBtn',...
    'setappdata(gcbf,''canceling'',1)');
setappdata(h,'canceling',0);
try
    while (CNT/TotalCNT < 1)
        if getappdata(h,'canceling')
            break
        end
        waitbar(CNT/TotalCNT,h,sprintf('%d / %d',CNT, TotalCNT));
        pause(1/ExtractFeaturesPerSecond);
    end
    delete(h);
    % Clean up everything
    stop(TimerData);
    delete(TimerData); 
    disp('Properly terminated');
catch
    % Clean up everything
    stop(TimerData);
    delete(TimerData);
    disp('Not properly terminated');
end
clear ExtractFeatures;

xysum = double([(FEATURE(1,:) + FEATURE(2,:)); (FEATURE(1,:) + FEATURE(3,:)); (sum(FEATURE))]);
x = min(512,max(1,round(xysum(1,:)./xysum(3,:)*511)));
y = min(512,max(1,round(xysum(2,:)./xysum(3,:)*511)));
sp = sparse(y, x, 1, 512, 512);
im = full(sp);
figure(6), subplot(1,2,1), imagesc(im');
figure(6), subplot(1,2,2), mesh(sp);

filename = sprintf('%s',FILENAME);
if CNT/TotalCNT == 1
    i = 1;
    while true
        if exist([filename '.mat'],'file') == 2
            filename = sprintf('%s (%d)',FILENAME,i);
            i = i + 1;
        else
            save([filename '.mat'],'FEATURE');
            disp('File saved!!');
            msgbox({'File saved!!' [filename '.mat']});
            break;
        end;
    end;
else
    disp('Canceled, Not saved!!');
    msgbox({'Canceled, Not saved!!' [filename '.mat']});
end;





% This function is called by the timer to display one frame of the figure
function ExtractFeatures(obj, event,ep00wire)
global FEATURE ELAPSE_TIME Stream_SIZE CNT TotalCNT;
tic;
% FPGA DAQ
numRows = int32(4); numCols = int32(Stream_SIZE);
% out = (Transfer_extract(numRows, numCols, ep00wire));
out = (Transfer_capture(numRows, numCols, ep00wire));
len = size(out,2);
if len ~= 1
    CNT = CNT + len;
    CNT = min(TotalCNT, CNT);
    out = out(:,11:numCols-10);
    FEATURE = [FEATURE out]; % 11:numCols-10);
    disp(size(FEATURE));
else
    disp('fifo empty');
end
ELAPSE_TIME(end+1) = toc;



% --- Executes during object creation, after setting all properties.
function txt_PSt_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





% --- Executes during object creation, after setting all properties.
function txt_Thr_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





% --- Executes during object creation, after setting all properties.
function txt_Csize_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





% --- Executes during object creation, after setting all properties.
function txt_TotalCnt_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end






function txt_filename_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end






% --- Executes during object creation, after setting all properties.
function txt_LLD_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function txt_ULD_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% % --- Executes on button press in btn_Pulse.
% function btn_Pulse_Callback(hObject, eventdata, handles)
% global FILENAME out CNT TotalCNT Cap_SIZE TEMP PULSE_Size;
% numPS = int32(PULSE_Size);
% ep00wire = TEMP;
% CNT = 0;
% Pulse = [];
% filename = FILENAME;
% i = 1;
% 
% % Define frame rate  
% h = waitbar(0,'1','Name', 'feature extracting...',...
%     'CreateCancelBtn',...
%     'setappdata(gcbf,''canceling'',1)');
% setappdata(h,'canceling',0);
% while (CNT/TotalCNT < 1)
%     if getappdata(h,'canceling')
%         break
%     end
%     numRows = int32(5); numCols = int32(Cap_SIZE);
%     pulse_vector = {};
%     try
%         [out, pulse_vector{1}, pulse_vector{2}, pulse_vector{3}, pulse_vector{4}] = (Transfer_pulse(numRows, numCols, ep00wire, numPS));% (Transfer_pulse(numRows, numCols, ep00wire));
%     catch
%         break;
%     end
%     len = size(out,2);
%     if len ~= 1
%         valid_index = 1:find(pulse_vector{1}(1,:),1,'last');
%         pulse_vector{1} = pulse_vector{1}(:,valid_index);
%         pulse_vector{2} = pulse_vector{2}(:,valid_index);
%         pulse_vector{3} = pulse_vector{3}(:,valid_index);
%         pulse_vector{4} = pulse_vector{4}(:,valid_index);
%         Pulse = [Pulse [pulse_vector{1}; pulse_vector{2}; pulse_vector{3}; pulse_vector{4}]];
%         disp(size(Pulse,2));
%         CNT = size(Pulse,2);
%         CNT = min(TotalCNT, CNT);
%     else
%         disp('fifo empty');
%     end
%     waitbar(CNT/TotalCNT,h,sprintf('%d / %d',CNT, TotalCNT));
%     % pause(1/ExtractFeaturesPerSecond);
% end
% try
%     delete(h);
%     disp('Properly terminated');
% catch
%     disp('Not properly terminated');
% end
% % Clean up everything
% 
% 
% while true
%     if exist([filename '.mat'],'file') == 2
%         filename = sprintf('%s (%d)',FILENAME,i);
%         i = i + 1;
%     else
%         save([filename '.mat'],'Pulse');
%         disp('File saved!!');
%         msgbox({'File saved!!' [filename '.mat']});
%         break;
%     end;
% end;






% % --- Executes on button press in btn_Save.
% function btn_Save_Callback(hObject, eventdata, handles)
% global FILENAME out CNT TotalCNT Cap_SIZE TEMP PULSE_Size;
% numPS = int32(PULSE_Size);
% ep00wire = TEMP;
% CNT = 0;
% Capture = [];
% filename = FILENAME;
% i = 1;
% 
% % Define frame rate  
% h = waitbar(0,'1','Name', 'feature extracting...',...
%     'CreateCancelBtn',...
%     'setappdata(gcbf,''canceling'',1)');
% setappdata(h,'canceling',0);
% while (CNT/TotalCNT < 1)
%     if getappdata(h,'canceling')
%         break
%     end
%     numRows = int32(5); numCols = int32(Cap_SIZE);
%     pulse_vector = {};
%     [out, pulse_vector{1}, pulse_vector{2}, pulse_vector{3}, pulse_vector{4}] = (Transfer_capture(numRows, numCols, ep00wire, numPS));% (Transfer_pulse(numRows, numCols, ep00wire));
%     len = size(out,2);
%     if len ~= 1
%         CNT = CNT + len;
%         CNT = min(TotalCNT, CNT);
%         out = out(:,1:numCols-1);
%         Capture = [Capture out];
%         disp(size(Capture));
%     else
%         disp('fifo empty');
%     end
%     waitbar(CNT/TotalCNT,h,sprintf('%d / %d',CNT, TotalCNT));
%     % pause(1/ExtractFeaturesPerSecond);
% end
% delete(h);
% % Clean up everything
% disp('Properly terminated');
% 
% while true
%     if exist([filename '.mat'],'file') == 2
%         filename = sprintf('%s (%d)',FILENAME,i);
%         i = i + 1;
%     else
%         save([filename '.mat'],'Capture');
%         disp('File saved!!');
%         msgbox({'File saved!!' [filename '.mat']});
%         break;
%     end;
% end;













% 
% function realVideo()
% global IM OUT handlesPlot S TEMP ELAPSE_TIME;
% ELAPSE_TIME = [];
% % Define frame rate  
% NumberFrameDisplayPerSecond=10;
% 
% % Open figure
% hFigure=figure(1); 
% 
% try
%    % For windows
%    vid = videoinput('winvideo',1, 'YUY2_320x240');
% catch
%    try
%       % For macs.
%       vid = videoinput('macvideo', 1);
%    catch
%       errordlg('No webcam available');
%    end
% end 
% 
% % Set parameters for video
% % Acquire only one frame each time
% set(vid,'FramesPerTrigger',1);
% % Go on forever until stopped
% set(vid,'TriggerRepeat',Inf); 
% % Get a grayscale image 
% set(vid,'ReturnedColorSpace','rgb');%'grayscale');
% triggerconfig(vid, 'Manual');
% 
% % FPGA Initialization
% % helloMex();
% 
% % set up timer object
% TimerData=timer('TimerFcn', {@FrameRateDisplay,vid,TEMP},'Period',1/NumberFrameDisplayPerSecond,'ExecutionMode','fixedRate','BusyMode','drop');
% 
% start(vid);
% start(TimerData); 
%  
% % We go on until the figure is closed
% try
%     uiwait(hFigure); 
%     % Clean up everything
%     stop(TimerData);
%     delete(TimerData); 
%     stop(vid);
%     delete(vid);
%     disp('Properly terminated');
% catch
%     % Clean up everything
%     stop(TimerData);
%     delete(TimerData);
%     stop(vid);
%     delete(vid);
%     disp('Not properly terminated');
% end
% clear FrameRateDisplay;
% 
% % 마지막 상태 출력
% % if first execution, we create the figure objects
% S = sparse(IM{3});
% figure(1),subplot(2,2,1);
% % Webcam Image
% handlesPlot{1}=imshow(IM{1}); % Webcam
% title('Webcam Image');
% % Gamma Image
% figure(1),subplot(2,2,2);
% handlesPlot{2}=imagesc(IM{2}); % imshow(IM{2}); % gamma Image
% title('Gamma Image');
% % Display X,Y,SUM
% figure(1),subplot(2,2,3);
% handlesPlot{3} = imagesc(IM{3}); % Display XYSUM
% title('Flood Image'); 
% % peakABCD
% figure(1),subplot(2,2,4); 
% handlesPlot{4} = mesh(S)'; % peakABCD
% title('3D flood histogram');
% xlabel('x'); 
% ylabel('y'); 
% zlabel('count');
% 
% 
% 
% 
% 
% 
% 
% % This function is called by the timer to display one frame of the figure
% function FrameRateDisplay(obj, event,vid,ep00wire)
% global IM OUT handlesPlot peakABCD ELAPSE_TIME S Stream_SIZE SIZX SIZY;
% tic;
% % FPGA DAQ  
% numRows = int32(3); numCols = int32(Stream_SIZE);
% OUT = (Transfer_realtime(numRows, numCols, ep00wire)); 
% disp([num2str(size(OUT)) ': ' num2str(sum(OUT(:)))]); 
% 
% % //position calculate (SCD)
% % double image_size = 1000;
% % double total = Math.Max(0.1, Math.Abs(data1[i] + data2[i] + data3[i] + data4[i]));
% % double xx = Math.Round(    image_size / 2 + (image_size / 2) * 3 * (data3[i] - data4[i]) / total  );
% % double yy = Math.Round(    image_size / 2 + (image_size / 2) * 3 * (data1[i] - data2[i]) / total  );
% 
% % Initialization 
% if isempty(IM)
%     IM = cell(1,3);
% end
% if isempty(handlesPlot)
%     handlesPlot = cell(1,4);
% end
% if isempty(IM{2})
%     IM{2} = zeros(512, 512);
%     IM{3} = zeros(512, 512);
%     [SIZX, SIZY] = size(IM{3}(:,:,1));
%     S = sparse(IM{3});
% end
% % OUT = [min(511,max(1,SIZY/2 - OUT(1,:))); min(511,max(1,SIZX/2 + OUT(2,:))); OUT(3:size(OUT,1),:)];
% trigger(vid);
% IM{1}=getdata(vid,1,'uint8');
% if size(OUT,2) == 1
%     trash = toc;
%     if isempty(handlesPlot{1}) == 0
%         set(handlesPlot{1},'CData',IM{1}); % Webcam
%     end
%     return 
% end 
% 
% IM{2} = rot90(full(sparse(double(OUT(2,:)+1),double(OUT(1,:)+1),1,SIZX,SIZY))',2);
% IM{3} = IM{3} + IM{2};
% if isempty(handlesPlot{1})
%    % if first execution, we create the figure objects
%    figure(1),subplot(2,2,1);
%    % Webcam Image
%    handlesPlot{1}=imshow(IM{1}); % Webcam
%    title('Webcam Image');
%    % Gamma Image
%    figure(1),subplot(2,2,2);
%    handlesPlot{2}=imagesc(IM{2}); % imshow(IM{2}); % gamma Image
%    title('Gamma Image');
%    % Display X,Y,SUM
%    figure(1),subplot(2,2,3);
%    handlesPlot{3} = imagesc((IM{3})); % Display XYSUM
%    title('Flood Image'); 
%    % peakABCD
%    figure(1),subplot(2,2,4);
%    handlesPlot{4} = mesh(S); % peakABCD
%    title('3D flood histogram');
%    xlabel('x'); 
%    ylabel('y'); 
%    zlabel('count');
%    
%    % peakABCD = OUT(4:5,:)'; % peakABCD and peakCnt
%    trash = toc;
% else
%    % XYSUM = num2cell(OUT(1:3,:),2); % XYSUM cell type
%    % We only update what is needed 
%    set(handlesPlot{1},'CData',IM{1}); % Webcam
%    set(handlesPlot{2},'CData',IM{2}); % gamma Image
%    set(handlesPlot{3},'CData',IM{3}); % Display XYSUM
%    set(handlesPlot{4},'ZData',IM{3}'); % Display XYSUM
%    % set(handlesPlot{4},'YData', OUT(1:3,:)); % peakABCD
%    
%    % peakABCD = [peakABCD; OUT(4:5,:)']; % peakABCD and peakCnt
%    ELAPSE_TIME(end+1) = toc;
% end
% 
% 
% 
% 
% 
% 
% 
