clear

addpath('/home/lcy/caffe/matlab');
test_data_pathadc = '/home/lcy/WORKSPACE/CAM/ADCCROPTJ';
%test_data_patht2 = '/home/lcy/WORKSPACE/CAM/DATAMIA/T2CROP/';
dirname=dir(fullfile(test_data_pathadc,'*.png'));
% load('labels.mat');
labels(1:264,1) = -1;


number = size(labels,1);
predictions = [];

%load the CAM model and extract features

net_weights = ['/home/lcy/WORKSPACE/CAM/models/ADCT2DUALModel/model/adc_t2_dua_CAM_cov_loss_relu_1_1_0.001/dual_adc_t2_CAM_cov_googlenet_iter_9500_relu_1_1_0.001.caffemodel'];
net_model = ['/home/lcy/WORKSPACE/CAM/models/Dualtrain/dual_CAM_cov_loss_adc_relu_deploy.prototxt'];
net = caffe.Net(net_model, net_weights, 'test');     

for i = 1:number
    i
    %adc
    imnameadc = fullfile(test_data_pathadc,dirname(i).name);
    img1adc = imread(imnameadc);
    imgadc=zeros(size(img1adc,1),size(img1adc,2),3);
    if size(img1adc,3)==1
        imgadc(:,:,1)=img1adc;
        imgadc(:,:,2)=img1adc;
        imgadc(:,:,3)=img1adc;
    else
        imgadc=img1adc;
    end
        
    imgadc = imresize(imgadc, [256 256]);
    
    %t2
%     imnamet2 = fullfile(test_data_patht2,dirname(i).name);
%     img1t2 = imread(imnamet2);
%     imgt2=zeros(size(img1t2,1),size(img1t2,2),3);
%     if size(img1t2,3)==1
%         imgt2(:,:,1)=img1t2;
%         imgt2(:,:,2)=img1t2;
%         imgt2(:,:,3)=img1t2;
%     else
%         imgt2=img1t2;
%     end
%         
%     imgt2 = imresize(imgt2, [256 256]);
    
    scores = net.forward({prepare_image(imgadc)});% extract conv features online
	scores = scores{1};
    scoresMean = mean(scores,4)';
    predictions = [predictions;scoresMean];  
end

predictf=[]
accuracy=0;
th=0;

for j=0.1:0.001:0.5
    z=0;
    predict=[];
    fn=find(predictions<j);
    fp=find(predictions>=j);
    predict(fn,1)=-1;
    predict(fp,1)=1; 
    predict(:,2)=predictions;
    predict(:,3)=labels;
    z1=predict(:,1)-predict(:,3);
    predict(:,4)=z1;
    ac1=numel(find(z1==0))./numel(labels);
    if accuracy<=ac1
        accuracy=ac1;
        th=j;
        predictf=predict;
    end   
end
