#References:-
#(1) A Neural Algorithm of Artistic Style - Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
#(2) Covoloutional Neural Networks by Andrew N.G  "https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF"
#(3) Neural Style Transfer using Keras "https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py"

from __future__ import print_function
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

from keras.applications import vgg19
from keras import backend as K

#Assign hyperparamters and intialize data
content_image_path='japanese_garden.jpg'
style_image_path='picasso_selfportrait.jpg'
result_name='japanese_garden_by_picasso'
iterations=6
content_weight_parameter=1*pow(10,-9)
style_weight_parameter=4*pow(10,-8)
total_variation_weight=4*pow(10,-8)
total_variation_loss_factor=1.25

#make dimensions of the pictures to be generated
width, height = load_img(content_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

#function to preprocess images into tensors
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

#function to deprocess tensors into image
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#Preprocess the content and style images to tensors
content_image = K.variable(preprocess_image(content_image_path))
style_image = K.variable(preprocess_image(style_image_path))

#create and store random generated image
if K.image_data_format() == 'channels_first':
    generated_image = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    generated_image = K.placeholder((1, img_nrows, img_ncols, 3))
	
#Combine the Content,Style,Generated image to push into the VGG19 model at one go
input_tensor = K.concatenate([content_image,
                              style_image,
                              generated_image], axis=0)

#Load the VGG19 model
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)

#Store the activations of the layers in a dictionary to utilize them at later time 					
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])					

#define the Content loss function
def content_loss(content, generated):
    return K.sum(K.square(generated - content))

#Compute the gram matrix function as given in "A Neural Algorithm of Artistic Style"
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

#define the style loss function with the help of gram matrix function
def style_loss(style,generated):
    assert K.ndim(style) == 3
    assert K.ndim(generated) == 3
    S = gram_matrix(style)
    C = gram_matrix(generated)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))	

#define the total variation loss function to keep image locally coherent
#This function si used to denoise the image
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, total_variation_loss_factor))

#Compute the total loss using the functions
#Content loss
#block4_con2 is Layer used to compute content loss
#Middle layers are more suitable for Content reproduction 
loss = K.variable(0.0)
layer_features = outputs_dict['block4_conv2']
content_image_features = layer_features[0, :, :, :]
generated_image_features = layer_features[2, :, :, :]
loss = loss + content_weight_parameter * content_loss(content_image_features,generated_image_features)
loss = loss + total_variation_weight * total_variation_loss(generated_image)

#Style loss
#Style loss is to be computed for the layers Layer1_1,Layer2_1,Layer3_1,Layer4_1,Layer5_1
feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    generated_image_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, generated_image_features)
    loss = loss + (style_weight_parameter / len(feature_layers)) * sl 	

#get the gradients with respect to the loss of the generated image
grads = K.gradients(loss, generated_image)
outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([generated_image], outputs)

#evaluate loss and gradient together and use the values for respective functions
def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

#define the evaluator class and loss and gradient functions
class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values											

#create object of evaluator
evaluator = Evaluator()

#run scipy-based optimization (L-BFGS) over the pixels of the generated image instead of the gradient descent as given in the reference paper
x = preprocess_image(content_image_path)
for i in range(iterations):
    print('Iteration', i+1)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save generated image
    img = deprocess_image(x.copy())
    fname ='iteration_%d.png' % (i+1)
    save_img(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % ((i+1), end_time - start_time))

save_img(result_name+'.png',img)		
