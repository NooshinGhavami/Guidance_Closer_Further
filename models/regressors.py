import torch.nn as nn
import numpy as np
import torch

# Network Architecture. Similar to the paper mentioned above which we are basing this on, we will have both shared
# layers and view-specific layers. However to start with we will have just one network all shared layers, and if this
# isn't working well then, similar to the paper, we can have the view-specific layers too.
# First create the layer definitions
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential( # Allows creation of sequentially ordered layers
            nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2), # create set of convolutional filters
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # we use stride of 2 because we want to downsample the image
        self.layer2 = nn.Sequential( # This is the second layer now
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(  # This is the third layer now
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #self.layer4 = nn.Sequential(  # This is the third layer now
         #   nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
          #  nn.BatchNorm2d(128),
           # nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2))
        #self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(32 * 32 * 64, 32) # The in_features x output_fetaures
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(32, 1)
        #self.dropout2 = nn.Dropout(p=0.3)
        self._initialise_weights() #call initialisation

    def _initialise_weights(self): # this function will initialise the weights of all the convolutional layers and the batch norms.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    # Now we define how the data flows through the layers in the forward pass of the network
    def forward(self, x):
        out = self.layer1(x)  # feed the input to layer1 (also permute the dimensions since it is not correct at the moment)
        out = self.layer2(out)  # feed the output of layer1 to the input of layer2
        out = self.layer3(out)  # feed the output of layer2 to the input of layer3
        #out = self.layer4(out)  # feed the output of layer2 to the input of layer4
        out = out.reshape(out.size(0), -1)  # flatten the data into a 1D vector before feeding into the fully connected (we want the first dimension as the batch size and it to figure out the other dimension)
        out = self.fc1(out)  # finally apply a fully connected layer to this
        out = self.dropout1(out)  # finally apply a fully connected layer to this
        out = self.fc2(out)  # finally apply a fully connected layer to this
        #out = self.dropout2(out)  # finally apply a fully connected layer to this
        return out  # Return this output from the function



class Regressor(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        # This should be in spherical coordinates
        self.input_size = tuple(input_size)

        # conpute conv paddings_cov
        kernels = (3,3)#(3, 3, 3, 3, 3)
        strides = (2,2)#(2, 2, 2, 2, 2)
        channels = (8,16)#(8, 16, 32, 32, 32)
        layer_size = (self.input_size,)
        for i in range(len(kernels)):
            layer_size += (tuple(int(np.ceil(l/strides[i])) for l in layer_size[i]),)

        padding = ()
        for i in range(len(kernels)):
            padding += (tuple(int(np.floor((l2*strides[i] - l+1*(kernels[i]-1))/2.0)) for l,l2 in zip(layer_size[i], layer_size[i+1])),)

        # output_padding = ()
        # for i in range(len(kernels)):
        #     output_padding += (tuple(l - (l2 - 1) * strides[i] + 2 * p - 1 * (kernels[i] - 1) - 1 for l,l2, p in zip(layer_size[i], layer_size[i+1], padding[i])),)

        channels = (2,) + channels # add input channel

        self.layer_size = layer_size
        #print(self.layer_size)

        cnn_layers = []
        for i in range(len(kernels)):
            current_layer = nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=kernels[i], stride=strides[i], padding=padding[i]),
                nn.ReLU(),
            )
            cnn_layers.append(current_layer)
        self.cnn = nn.Sequential(*cnn_layers)

        n_features_last_cnn_layer = np.prod(self.layer_size[-1])*channels[-1]

        self.linear_layers = nn.Sequential(
            nn.Linear(n_features_last_cnn_layer, 64),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(64, 2),
            #nn.Sigmoid()
        )

    def forward(self, data):
        batch_size = data.shape[0]
        a = self.cnn(data)
        a = a.view(batch_size,-1)
        y = self.linear_layers(a)
        return y
