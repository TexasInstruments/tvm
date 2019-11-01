# This script is intended for execution on AM57
# In same folder, deploy_* artifacts generated on TVM x86 side should be placed.
# Also, in this same folder inference configurarion file should be accessible
import tvm
import tvm
import os
import cv2
import numpy as np
from tvm.contrib import graph_runtime

data_shape_input = [1,224,224,3]
#data = np.random.uniform(-1, 1, size=data_shape_input).astype("float32")

#image_file = "../test/testvecs/input/objects/cat-pet-animal-domestic-104827.jpeg"
#image_file = "../classification/images/plane.jpg"
image_file = "../classification/images/tennis_ball.jpg"

# image dimension to be obtained from input tensor shape
input_dim = tuple(data_shape_input[1:3])
# openCV reads image as BGR format
image_BGR = cv2.imread(filename = image_file)
# convert to RGB format
image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

# crop and resize image
orig_H = image.shape[0]
orig_W = image.shape[1]
print("InputH=" + str(orig_H) + " InputW=" + str(orig_W))
if(orig_H < orig_W):
  x0f = np.floor((orig_W - orig_H) / 2)
  x1f = orig_W - x0f
  y0f = 0
  y1f = orig_H
else:
  x0f = np.floor((orig_H - orig_W) / 2)
  x1f = orig_H - x0f
  y0f = 0
  y1f = orig_W

x0 = np.floor(x0f).astype('int')
x1 = np.floor(x1f).astype('int')
y0 = np.floor(y0f).astype('int')
y1 = np.floor(y1f).astype('int')

cropped_image = image[x0:x1,y0:y1]
print("CroppedImage:" + str(x0) + "," + str(y0) + "   " + str(x1) + "," + str(y1))
resized_image = cv2.resize(cropped_image, input_dim, interpolation = cv2.INTER_AREA)
np_image_data = np.asarray(resized_image)
np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.99, .99, cv2.NORM_MINMAX)
print("np_image_data shape:" + str(np_image_data.shape))
data = np.expand_dims(np_image_data, 0)
print("data shape:" + str(data.shape))

# load the module back.
tmp_folder = os.getcwd() + "/"
path_lib = tmp_folder + "deploy_lib.tar"
path_graph = tmp_folder + "deploy_graph.json"
path_params = tmp_folder + "deploy_param.params"

loaded_json = open(path_graph).read()
loaded_lib = tvm.module.load(path_lib)
loaded_params = bytearray(open(path_params, "rb").read())

# Create graph runtime
module = graph_runtime.create(loaded_json, loaded_lib, ctx=tvm.cpu(0))
module.load_params(loaded_params)
#Set the input using normalized NHWC data (prepared earlier)
module.set_input("x", data)
# Execute inference, from TVM-RT->TIDL-API->OpenCL->TIDL
module.run()
# Get output tensor
out = module.get_output(0).asnumpy()
# Print input data
#print(data)
# Print output tensor
np.set_printoptions(threshold=np.inf)
print(out)
# Show max value and its index, classification result
print("MAX VALUE:" + str(np.amax(out)) + " at index:" + str(np.argmax(out)))

