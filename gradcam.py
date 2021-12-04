import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch.autograd import Function
from skimage import transform


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            # Considering the 3rd depth of Encoder3D
            #if name == 'maxpool':
            #    x = module(x)
            #    continue
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class GradExtractor():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targetted layers.
    3. Gradients from intermediate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients  # return gradient

    def __call__(self, x):  # return conv_output & model_output
        conv_output = []
        for name, module in self.model._modules.items():
            #print(name)
            if module == self.feature_module:
                conv_output, x = self.feature_extractor(x)
            elif "block" in name:
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)
            #print(x.shape)

        return conv_output, x


class GradCam:
    def __init__(self, model, feature_module, target_layer_names):
        self.model = model.eval()
        self.feature_module = feature_module
        self.model = model.cuda()
        self.extractor = GradExtractor(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input):  # input: [1,C,H,W]
               # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor(input)
        target_class = np.argmax(model_output.data.cpu().numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.feature_module.zero_grad()
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output.cuda(), retain_graph=True)
        # Get hooked gradient
        gradient = self.extractor.get_gradients()[0].cpu().data.numpy()  # [1,C,H,W,D]
        # Get weights from gradients, take averages for each gradient
        weights = np.mean(gradient, axis=(2, 3, 4))[0, :]  # [C]
        # Get convolution outputs
        feature = conv_output[-1]
        feature = feature.cpu().data.numpy()[0, :]  # [C,H,W,D]
        # Create empty numpy array for cam
        cam = np.zeros(feature.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        featurevalue = []
        for i, w in enumerate(weights):
           cam += w * feature[i, :, :, :]
           featurevalue.append(feature[i, :, :, :].flatten().mean())
        # equivalent to ReLU, set 0 value for those negative  # [H,W,D]
        cam = np.maximum(cam, 0)
        # upsample to the origin image size
        cam = transform.resize(cam, input.shape[2:])
        # scale between 0-1
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam # feature[0, :, :, :]


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


