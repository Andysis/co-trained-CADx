# -*- coding: utf-8 -*-  

#import sys
#sys.path.insert(0,'/workspace/lcy/caffe/python')
import caffe
import matplotlib.pyplot as plt  

import numpy as np
import os



#weights = '/home/lcy/WORKSPACE/CAM/models/ADCT2DUALModel/model/adc_t2_dua_CAM_cov/dual_adc_t2_CAM_cov_googlenet_iter_9500.caffemodel'


# init
#caffe.set_device(int(sys.argv[1]))
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('/home/lcy/WORKSPACE/CAM/models/Dualtrain/solverCAM_cov_loss_min.prototxt')
#solver.net.copy_from(weights)


# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#surgery.interp(solver.net, interp_layers)

#train
#solver.solve()

# scoring
#val = np.loadtxt('/workspace/lcy/fcn/noduledata/valname.txt', dtype=str)

#for _ in range(25):
#    solver.step(4000)
#    score.seg_tests(solver, False, val, layer='score')

niter = 50000
display= 5
#test_iter = 1
#test_interval = 50

train_lossadc = np.zeros(np.ceil(niter * 1.0 / display))   
train_losst2 = np.zeros(np.ceil(niter * 1.0 / display))
sim_loss = np.zeros(np.ceil(niter * 1.0 / display))      
  

# iteration 0  
solver.step(1)  
  
# 辅助变量  
_train_lossadc = 0; _train_losst2 = 0; _sim_loss = 0;  
# 进行解算  
for it in range(niter):  
    # 进行一次解算  
    solver.step(1)  
    reshapeadc = solver.net.blobs['sigadc'].data
    reshapet2 = solver.net.blobs['sigt2'].data
    sim = solver.net.blobs['sim'].data
    z=np.zeros((2,196))
    z[0,:]=reshapeadc[0,:,0,0]
    z[1,:]=reshapet2[0,:,0,0]
    #print sim 
    #print z[0]
    #print z[1] 
    #print reshapeadc[0,5,:,:] 
    #print reshapet2[0,5,:,:] 
    # 每迭代一次，训练batch_size张图片  
    _train_lossadc += solver.net.blobs['lossadc'].data  
    _train_losst2 += solver.net.blobs['losst2'].data 
    _sim_loss += solver.net.blobs['simloss'].data
    if it % display == 0:  
        # 计算平均train loss  
        train_lossadc[it // display] = _train_lossadc / display
        train_losst2[it // display] = _train_losst2 / display    
        sim_loss[it // display] = _sim_loss / display    
        _train_lossadc = 0
        _train_losst2 = 0 
        _sim_loss = 0 
  
    #if it % test_interval == 0:  
       # for test_it in range(test_iter):  
            # 进行一次测试  
        #    solver.test_nets[0].forward()  
            # 计算test loss  
            #_test_loss += solver.test_nets[0].blobs['loss'].data  
            # 计算test accuracy  
            #_accuracy += solver.test_nets[0].blobs['accuracy'].data  
        # 计算平均test loss  
        #test_loss[it / test_interval] = _test_loss / test_iter  
        # 计算平均test accuracy  
      #  test_acc[it / test_interval] = _accuracy / test_iter  
      #  _test_loss = 0  
      #  _accuracy = 0  
  
# 绘制train loss、test loss和accuracy曲线  
print '\nplot the train loss and test accuracy\n'  
fig, ax1 = plt.subplots()  
  
  
# train loss -> 绿色  
ax1.plot(display * np.arange(len(train_lossadc)), train_lossadc, 'g')  
# test loss -> 黄色  
ax1.plot(display * np.arange(len(train_losst2)), train_losst2, 'y')  

fig1, ax2 = plt.subplots()  

# test accuracy -> 红色  
ax2.plot(display * np.arange(len(sim_loss)), sim_loss, 'r')  
  
ax1.set_xlabel('iteration')  
ax1.set_ylabel('loss')
ax2.set_xlabel('iteration')  
ax2.set_ylabel('loss')  
#plt.show()  
fig.savefig('/home/lcy/WORKSPACE/CAM/models/ADCT2DUALModel/model/scratch.png')
fig1.savefig('/home/lcy/WORKSPACE/CAM/models/ADCT2DUALModel/model/scratch.png')