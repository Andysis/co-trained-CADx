% extract feature from models

addpath('/home/lcy/caffe/matlab');

% load model
net_weights = ['/home/lcy/WORKSPACE/CAM/models/ADCT2DUALModel/model/dual_adc_t2_CAM_cov_googlenet_iter_9500_relu_1_1_0.001.caffemodel'];
%adc model
net_modeladc = ['/home/lcy/WORKSPACE/CAM/models/Dualtrain/dual_CAM_cov_loss_adc_relu_deploy.prototxt'];
netadc = caffe.Net(net_modeladc, net_weights, 'test');  
%t2 model
net_modelt2 = ['/home/lcy/WORKSPACE/CAM/models/Dualtrain/dual_CAM_cov_loss_t2_relu_deploy.prototxt'];
nett2 = caffe.Net(net_modelt2, net_weights, 'test');  

data_path_adc = '/home/lcy/WORKSPACE/CAM/DATAMIA/SVMDATA/ADCTRAIN/ids6/';
data_path_t2 = '/home/lcy/WORKSPACE/CAM/DATAMIA/SVMDATA/T2TRAIN/ids6/';
dirname = dir(fullfile(data_path_adc,'*.png'));
write_path='/home/lcy/WORKSPACE/CAM/DATAMIA/SVMDATA/Feature/';
featureadc = [];
featuret2 = [];
for i = 1:numel(dirname)
    i
    %adc
    imnameadc=fullfile(data_path_adc,dirname(i).name);
    img1adc=imread(imnameadc);
    imgadc=zeros(221,221,3);
    if size(img1adc,3)==1
        imgadc(:,:,1)=img1adc;
        imgadc(:,:,2)=img1adc;
        imgadc(:,:,3)=img1adc;
    else
        imgadc=img1adc;
    end
    
    imgadc = imresize(imgadc,[224 224]); 
    scoresadc = netadc.forward({imgadc});
    feature1adc = netadc.blobs('CAM_convadc').get_data();   
    for j=1:1024
        feadc = feature1adc(:,:,j); 
        featureadc(i,j) = mean(feadc(:));
    end
    
    %t2 
    imnamet2=fullfile(data_path_t2,dirname(i).name);
    img1t2=imread(imnamet2);
    imgt2=zeros(size(img1t2,1),size(img1t2,2),3);
    if size(img1t2,3)==1
        imgt2(:,:,1)=img1t2;
        imgt2(:,:,2)=img1t2;
        imgt2(:,:,3)=img1t2;
    else
        imgt2=img1t2;
    end
    
    imgt2 = imresize(imgt2,[224 224]); 
    scorest2 = nett2.forward({imgt2});
    feature1t2 = nett2.blobs('CAM_convt2').get_data();   
    for j=1:1024
        fet2 = feature1t2(:,:,j); 
        featuret2(i,j) = mean(fet2(:));
    end
end

writenameadc=fullfile(write_path,'featureadctrains6.mat');
save(writenameadc,'featureadc');
writenamet2=fullfile(write_path,'featuret2trains6.mat');
save(writenamet2,'featuret2');

feature = [featureadc,featuret2];

writename = fullfile(write_path,'featuretrains6.mat');
save(writename,'feature');
    
    
    