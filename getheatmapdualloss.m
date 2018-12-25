clear
addpath('/home/lcy/caffe/matlab');
test_data_pathadc = '/home/lcy/WORKSPACE/CAM/ADCCROPTJ';
%test_data_patht2 = '/home/lcy/WORKSPACE/CAM/T2PTRAIN';
dirname=dir(fullfile(test_data_pathadc,'*.png'));
load('labels.mat');
number = size(labels,1);

% load the CAM model and extract features

net_weights = ['/home/lcy/WORKSPACE/CAM/models/ADCT2DUALModel/model/adc_t2_dua_CAM_cov_loss_relu_1_1_0.001/dual_adc_t2_CAM_cov_googlenet_iter_9500_relu_1_1_0.001.caffemodel'];
net_model = ['/home/lcy/WORKSPACE/CAM/models/Dualtrain/dual_CAM_cov_loss_adc_relu_deploy.prototxt'];
net = caffe.Net(net_model, net_weights, 'test');    
%weights_LR = net.params('CAM_fcx',1).get_data() ;% get the softmax layer of the network

for i = 1:numel(dirname)
%     i
    %adcheat
    imgIDadc = fullfile(test_data_pathadc,dirname(i).name)
    img1adc = imread(imgIDadc);
    imgadc = zeros(size(img1adc,1),size(img1adc,2),3);   
    imgadc(:,:,1) = img1adc(:,:,1);
    imgadc(:,:,2) = img1adc(:,:,1);
    imgadc(:,:,3) = img1adc(:,:,1);
    imgadc = imresize(imgadc, [256 256]);    
%     %t2
%     imgIDt2 = fullfile(test_data_patht2,dirname(i).name)
%     img1t2 = imread(imgIDt2);
%     imgt2 = zeros(size(img1t2,1),size(img1t2,2),3);
%     imgt2(:,:,1) = img1t2(:,:,1);
%     imgt2(:,:,2) = img1t2(:,:,1);
%     imgt2(:,:,3) = img1t2(:,:,1);
%     imgt2 = imresize(imgt2, [256 256]); 
    
%     padc = prepare_image(imgadc);
%     pt2 = prepare_image(imgt2);
%     pmix = zeros(size(padc,1),size(padc,2),2,10);
%     pmix(:,:,1,:) = padc(:,:,1,:);
%     pmix(:,:,2,:) = pt2(:,:,1,:);
    
    scores = net.forward({prepare_image(imgadc)});% extract conv features online
    activation_lastconv = net.blobs('CAM_convadc2').get_data();
    scores = scores{1};

    %% Class Activation Mapping

    topNum = 1; % generate heatmap for top X prediction results
    scoresMean = mean(scores,2);
    [value_category, IDX_category] = sort(scoresMean,'descend');
    %[curCAMmapAll] = returnCAMmap(activation_lastconv, weights_LR(:,IDX_category(1:topNum)));
    curCAMmapAll = activation_lastconv;
    
    
    curPrediction = '';
    curResult = im2double(imgadc);

    for j=1:topNum
        curCAMmap_crops = squeeze(curCAMmapAll(:,:,j,:));
        curCAMmapLarge_crops = imresize(curCAMmap_crops,[256 256]); 
        curCAMLarge = mergeTenCrop(curCAMmapLarge_crops);
        curHeatMap = imresize(im2double(curCAMLarge),[256 256]);
        curHeatMap = im2double(curHeatMap);
        imgmap = double(curHeatMap);
        
        %img = double(imgadc);
        %img = img./max(img(:));
        
        %curHeatMap = map2jpg(curHeatMap,[], 'jet');
       % curHeatMap = im2double(img)*0.6+curHeatMap*0.3;
        
        
          if(~exist('range', 'var') || isempty(range)), range = [min(imgmap(:)) max(imgmap(:))]; end
  
         heatmap_gray = mat2gray(imgmap, range);        
         imwrite(heatmap_gray, ['/home/lcy/WORKSPACE/CAM/TONGJIBPH/',[dirname(i).name(1:end-4),'.jpg']]);
         save(['/home/lcy/WORKSPACE/CAM/TONGJIBPH/',[dirname(i).name(1:end-4),'.mat']],'imgmap')
    end

    if mod(i,5) == 0
        fprintf('%d\n',i);
    end
end