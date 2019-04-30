from caffe import layers as L,params as P,to_proto
import matplotlib.pyplot as plt
import caffe


# 载入网络，列出各个层的名字
# net = caffe.Net('./pi-object-detection/MobileNetSSD_deploy.prototxt', caffe.TRAIN)
net = caffe.Net('./Mobie_TinySSD_test.prototxt', caffe.TEST)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

