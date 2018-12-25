clear
addpath('/home/lcy/caffe/matlab');


% load the CAM model and extract features

net_weights = ['/home/lcy/WORKSPACE/CAM/models/ADCT2DUALModel/model/adc_t2_dua_CAM_cov_loss_relu_1_1_0.001/dual_adc_t2_CAM_cov_googlenet_iter_9500_relu_1_1_0.001.caffemodel'];
net_model = ['/home/lcy/WORKSPACE/CAM/models/Dualtrain/dual_CAM_cov_loss_adc_relu_deploy.prototxt'];
net = caffe.Net(net_model, net_weights, 'test');       

%% read adc and t2 image
img1adc = imread('adc.png');
imgadc = zeros(size(img1adc,1),size(img1adc,2),3);
imgadc(:,:,1) = img1adc(:,:,1);
imgadc(:,:,2) = img1adc(:,:,1);
imgadc(:,:,3) = img1adc(:,:,1);
imgadc = imresize(imgadc, [256 256]);    

%% predict adc

scores = net.forward({prepare_image(imgadc)});% extract conv features online
activation_lastconv = net.blobs('CAM_convadc2').get_data();
scores = scores{1};


%% adc Class Activation Mapping

topNum = 1; % generate heatmap for top X prediction results
scoresMean = mean(scores,2);
curCAMmapAll = activation_lastconv;


curPrediction = '';
curPrediction_mix = '';
curResult = im2double(imgadc);

for j=1:topNum
    curCAMmap_crops = squeeze(curCAMmapAll(:,:,j,:));
    curCAMmapLarge_crops = imresize(curCAMmap_crops,[256 256]); 
    curCAMLarge = mergeTenCrop(curCAMmapLarge_crops);
    curHeatMap = imresize(im2double(curCAMLarge),[256 256]);
    curHeatMap = im2double(curHeatMap);
    imgmap = double(curHeatMap);

    img = double(imgadc);
    img = img./max(img(:));

    curHeatMap = map2jpg(curHeatMap,[], 'jet');
    curHeatMap = im2double(img)*0.6+curHeatMap*0.3;


      if(~exist('range', 'var') || isempty(range)), range = [min(imgmap(:)) max(imgmap(:))]; end

     heatmap_gray = mat2gray(imgmap, range);        
     
end

figure,imshow(heatmap_gray);title(curPrediction)
figure,imshow(curHeatMap);title(curPrediction_mix)

