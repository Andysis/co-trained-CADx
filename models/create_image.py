import caffe
import numpy as np



class create_rgb_image(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) !=1 :
            raise Exception("Need one inputs to convert image")

    def reshape(self, bottom, top):
        # accuracy output is scalar
        self.data = bottom[0].data
        #import pdb; pdb.set_trace()
        top[0].reshape( self.data.shape[0],3,self.data.shape[2],self.data.shape[3] )
        

    def forward(self, bottom, top):
        new_array = np.zeros((self.data.shape[0],3,self.data.shape[2],self.data.shape[3]))
        #import pdb; pdb.set_trace() 
        for i in range(self.data.shape[0]):
            new_array[i,0,:,:] = self.data[i,0,:,:] 
            new_array[i,1,:,:] = self.data[i,0,:,:]   
            new_array[i,2,:,:] = self.data[i,0,:,:]     
        
        top[0].data[...] = new_array
        #import pdb; pdb.set_trace()
        #print np.shape(top[0].data)
        
    def backward(self, top, propagate_down, bottom):
        # no back-prop for input layers
        pass