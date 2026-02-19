import torch
import torch.nn as nn
import torch.nn.functional as F


def val(model, val_data_loader, criterion, device):
    """
    Inputs:
    model (torch.nn.Module): The deep learning model to be trained.
    val_data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    criterion (torch.nn.Module): Loss function to compute the training loss.
    device (torch.device): The device (CPU/GPU) to run the evaluation on.

    Outputs:
    Validation loss
    """
    val_running_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_data_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            
    model.train()
    return val_running_loss/len(val_data_loader)


def train(model, data_loader, val_data_loader, criterion, optimizer, epochs, device):
    """
    Inputs:
    model (torch.nn.Module): The deep learning model to be trained.
    data_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    val_data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    criterion (torch.nn.Module): Loss function to compute the training loss.
    optimizer (torch.optim.Optimizer): Optimizer used for updating the model parameters.
    epochs (int): Number of training epochs.
    device (torch.device): The device (CPU/GPU) to run the evaluation on.

    Outputs:
    Tuple of (train_loss_arr, val_loss_arr), an array of the training and validation
    losses at each epoch
    """
    train_loss_arr = []
    val_loss_arr = []
    running_loss = 0.0

    for epoch in range(epochs): # for each epoch, train the whole data_loader
        # TODO: write a training loop
        model.train()
        running_loss = 0.0 # otherwise, the train loss will be increasing

        for data, target in data_loader: # iterate all the data and target in the data loader
          # move data and target to GPU
          data, target = data.to(device), target.to(device)

          optimizer.zero_grad()
          output = model(data)

          loss = criterion(output, target)
          loss.backward()
          optimizer.step()
        
          running_loss+=loss.item()
        
        train_loss = running_loss/len(data_loader)
        train_loss_arr.append(train_loss)

        validation_loss = val(model, val_data_loader, criterion, device)
        val_loss_arr.append(validation_loss)
        print(f"Epoch: {epoch}, train loss: {train_loss}, validation loss: {validation_loss}")
        # END TODO

    print("Training finished.")
    return train_loss_arr, val_loss_arr


class ConvNet(nn.Module):
    """
    A Convolutional Neural Net with the following layers
    - conv1: convolution layer with 4 output channels, kernel size of 3, stride of 2, padding of 1 (input is a color image)
    - ReLU nonlinearity
    - conv2: convolution layer with 16 output channels, kernel size of 3, stride of 2, padding of 1
    - ReLU nonlinearity
    - conv3: convolution layer with 32 output channels, kernel size of 3, stride of 2, padding of 1
    - ReLU nonlinearity
    - fc1:    fully connected layer with 1024 output features
    - ReLU nonlinearity
    - fc2:   fully connected layer with 4 output features (the number of classes)
    """

    def __init__(self, num_classes=4):
        super(ConvNet, self).__init__()
        # TODO: define the network
        # input: (1, 3, 64, 64) => C=3, H=64, W=64
        self.conv1 = nn.Conv2d(
          in_channels=3,
          out_channels=4,
          kernel_size=3,
          stride=2,
          padding=1
        )
        self.conv2 = nn.Conv2d(
          in_channels=4,
          out_channels=16,
          kernel_size=3,
          stride=2,
          padding=1
        )
        self.conv3 = nn.Conv2d(
          in_channels=16,
          out_channels=32,
          kernel_size=3,
          stride=2,
          padding=1
        )
        self.fc1 = nn.LazyLinear(out_features=1024)
        # self.fc1 = nn.Linear(in_features=32*4*4, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=4)
        # END TODO

    def forward(self, x):
        # TODO: create a convnet forward pass
        # nn.ReLU could be put in the __init__ function
        # but F.relu must be put in the forward function
        x = F.relu(self.conv1(x)) # [1, 4, 32, 32]
        x = F.relu(self.conv2(x)) # [1 16, 16, 16]
        x = F.relu(self.conv3(x)) # [1, 32, 8, 8] 32 images and each image is 8*8
        x = torch.flatten(x, 1) # [1, 2048]
        x = F.relu(self.fc1(x)) # [1, 1024]
        x = self.fc2(x) # [1, 4]
        # END TODO

        return x

class ConvNetMaxPooling(nn.Module):
    """
    Same as ConvNet but with added max pooling of stride = 2 and kernel = 2 after each convolutional block's ReLU
    """

    def __init__(self, num_classes=4):
        super(ConvNetMaxPooling, self).__init__()
        # TODO: define network
        self.conv1 = nn.Conv2d(
          in_channels=3,
          out_channels=4,
          kernel_size=3,
          stride=2,
          padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
          in_channels=4,
          out_channels=16,
          kernel_size=3,
          stride=2,
          padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
          in_channels=16,
          out_channels=32,
          kernel_size=3,
          stride=2,
          padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.LazyLinear(out_features=1024)
        # self.fc1 = nn.Linear(in_features=32*4*4, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=4)
        # END TODO

    def forward(self, x):
        # TODO: create a convnet forward pass
        x = self.pool1(F.relu(self.conv1(x))) 
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x))) 
        x = torch.flatten(x, 1) # [1, 2048]
        x = self.fc2(F.relu(self.fc1(x)))
        # END TODO
        return x


# Custom Batch Normalization implementation
class BatchNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Initialize a custom Batch Normalization module.

        INPUT:
        - num_features (int): The number of features (channels) to be normalized.
        - eps (float): A small value added to the denominator for numerical stability during normalization.
        - momentum (float): The momentum for updating running statistics.
        """
        super(BatchNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weights = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # We use register buffers so that these tensors move with the model to 
        # the correct device
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        """
        Forward pass of the custom BatchNormalization module.

        INPUT:
        - x (Tensor of shape (B, C, H, W)): The input tensor (representing a batch) to be normalized.

        OUTPUT:
        - x (Tensor of shape (B, C, H, W): The output tensor after normalization.
                        
        DIMENSIONS:
            - B (Batch size): The number of samples in the batch.
            - C (Channels): The number of feature channels (e.g., 3 for RGB images).
            - H (Height): The height of each image or feature map.
            - W (Width): The width of each image or feature map.

        These variable names (B, C, H, W) are commonly used conventions in PyTorch
        
        """
        # Hint: You need to understand broadcasting in Pytorch! https://pytorch.org/docs/stable/notes/broadcasting.html
        # The mean and variance vector should be of shape (1, C, 1, 1)
        if self.training:
            # TODO: Implement the logic for training
            # we need to normalize the Channel dimension, cause this dimension means different features
            batch_mean = torch.mean(x, dim=(0,2,3), keepdims=True)
            batch_var = torch.var(x, dim=(0,2,3), keepdims=True)

            # get the batch normalization of x
            x = (x-batch_mean)/torch.sqrt(batch_var+self.eps) 

            # broadcase self.weights and self.bias
            x = self.weights.view(1, -1, 1, 1)*x+self.bias.view(1, -1, 1, 1)
            # update running_mean and running_var
            with torch.no_grad():
              self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*batch_mean.view(-1) # (1, 3, 1, 3)
              self.running_var=(1-self.momentum)*self.running_var+self.momentum*batch_var.view(-1) # (1, 3, 1, 3)
            # END TODO
        else:
            # TODO: Implement the logic for inference
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
            x = (x-mean)/torch.sqrt(var+self.eps)
            x = self.weights.view(1, -1, 1, 1)*x+self.bias.view(1, -1, 1, 1)
            # END TODO
        return x


class ConvNetBN(nn.Module):
    """
    Same as ConvNetMaxPooling but with BatchNormalization layers after each convolution.
    """

    def __init__(self, num_classes=4):
        super(ConvNetBN, self).__init__()
        # TODO: define network
        self.conv1 = nn.Conv2d(
          in_channels=3,
          out_channels=4,
          kernel_size=3,
          stride=2,
          padding=1
        )
        self.bn1 = BatchNormalization(4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
          in_channels=4,
          out_channels=16,
          kernel_size=3,
          stride=2,
          padding=1
        )
        self.bn2 = BatchNormalization(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
          in_channels=16,
          out_channels=32,
          kernel_size=3,
          stride=2,
          padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = BatchNormalization(32)

        self.fc1 = nn.LazyLinear(out_features=1024)
        # self.fc1 = nn.Linear(in_features=32*4*4, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
        # END TODO

    def forward(self, x):
        # TODO: create a convnet forward pass
        # add BN layers after each convolutional layer
        x = self.pool1(F.relu(self.bn1(self.conv1(x)))) # [1, 4, 32, 32]
        x = self.pool2(F.relu(self.bn2(self.conv2(x)))) # [1 16, 16, 16]
        x = self.pool3(F.relu(self.bn3(self.conv3(x)))) # [1, 32, 8, 8] 32 images and each image is 8*8
        x = torch.flatten(x, 1) # [1, 2048]
        x = F.relu(self.fc1(x)) # [1, 1024]
        x = self.fc2(x) # [1, 4]
        # END TODO
        return x


class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        """
        Custom Dropout Layer Initialization.

        Parameters:
        - p (float): The dropout probability, which determines the fraction of
          input elements that will be set to zero during training. It should
          be a value between 0 and 1. By default, it is set to 0.5.

        Returns:
        - None
        """
        super(CustomDropout, self).__init__()
        self.p = p

    def forward(self, x):
        """
        Forward Pass of the Custom Dropout Layer.

        Parameters:
        - x (Tensor): The tensor of weights to which dropout will be applied.

        Returns:
        - x (Tensor): The modified input tensor after applying dropout.
        """
        if self.training:
            # TODO: Implement the training behavior of dropout.
            keep_prob = 1-self.p
            rand = torch.rand_like(x)
            mask = (rand>self.p).float()
            x = x*mask/(1-self.p)
            # END TODO
        else:
            # TODO: Implement the inference behavior of dropout.
            x = x # do nothing
            # END TODO
        return x


class ConvNetDropout(nn.Module):
    def __init__(self, num_classes=4):
        super(ConvNetDropout, self).__init__()
        # TODO: define network
        self.conv1 = nn.Conv2d(
          in_channels=3,
          out_channels=4,
          kernel_size=3,
          stride=2,
          padding=1
        )
        self.bn1 = BatchNormalization(4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(
          in_channels=4,
          out_channels=16,
          kernel_size=3,
          stride=2,
          padding=1
        )
        self.bn2 = BatchNormalization(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(
          in_channels=16,
          out_channels=32,
          kernel_size=3,
          stride=2,
          padding=1
        )
        self.bn3 = BatchNormalization(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.LazyLinear(out_features=1024)
        # self.fc1 = nn.Linear(in_features=32*4*4, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=4)

        # NOTE: whether we need to use our own dropout or not ?????
        self.drop1 = CustomDropout(p=0.5)
        self.drop2 = CustomDropout(p=0.5)
        self.drop3 = CustomDropout(p=0.5)
        self.drop4 = CustomDropout(p=0.5)
        # END TODO

    def forward(self, x):
        # TODO: create a convnet forward pass
        x = self.pool1(self.drop1(F.relu(self.bn1(self.conv1(x))))) # [1, 4, 32, 32]
        x = self.pool2(self.drop2(F.relu(self.bn2(self.conv2(x))))) # [1 16, 16, 16]
        x = self.pool3(self.drop3(F.relu(self.bn3(self.conv3(x))))) # [1, 32, 8, 8] 32 images and each image is 8*8
        x = torch.flatten(x, 1) # [1, 2048]
        x = self.drop4(F.relu(self.fc1(x))) # [1, 1024]
        x = self.fc2(x) # [1, 4]
        # END TODO
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, interm_channel, out_channel, stride=1):
        """
        Inputs:
        in_channel = number of channels in the input to the first convolutional layer
        interm_channel = number of channels in the output of the first convolutional layer
                       = number of channels in the input to the second convolutional layer
        out_channel = number of channels in the output
        stride = stride for convolution, defaults to 1
        """
        super().__init__()
        # 3 convolutional layers: named conv1, conv2, and conv3
        # 2 batch normalization layers: named bn1, bn2

        # TODO: initialize a residual block with the layers specified above
        self.conv1 = nn.Conv2d(
          in_channels=in_channel,
          out_channels=interm_channel,
          kernel_size=3,
          stride=stride,
          padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=interm_channel)
        self.relu = nn.ReLU()


        self.conv2 = nn.Conv2d(
          in_channels=interm_channel,
          out_channels=out_channel,
          kernel_size=3,
          stride=stride,
          padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)

        # use conv3 to adjusts the input tensor, which make the ouput dim matches the main branch out_channel
        self.conv3 = nn.Conv2d(
          in_channels=in_channel,
          out_channels=out_channel,
          kernel_size=1,
          stride=stride
        )
        # END TODO

    def forward(self, x):
        # TODO: implement the forward function based on the architecture above
        output1 = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        output2 = self.conv3(x)

        output = self.relu(output1+output2)
        # END TODO
        return output


class ResNet(nn.Module):
    def __init__(
        self, num_blocks, layer1_channel, layer2_channel, out_channel, num_classes=4
    ):
        """
        Inputs:
        num_blocks = number of blocks in a block layer
        layer1_channel = number of channels in the input to the first block layer
        layer2_channel = number of channels in the output of the first block layer
                       = number of channels in the input to the second blcok layer
        out_channel = number of channels in the output
        num_classes = number of classes in the classification output, defaults to 4
        """
        super(ResNet, self).__init__()
        self.first = nn.Sequential(
            nn.LazyConv2d(layer1_channel, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d(
                (1, 1)), nn.Flatten(), nn.LazyLinear(num_classes)
        )
        # initialize two layers called layer1 and layer2 with num_blocks residual blocks each.

        # TODO: (important!) implement the block layer function below before this part
        self.layer1 = self.block_layer(num_blocks=num_blocks, in_channel=layer1_channel, out_channel=layer2_channel)
        self.layer2 = self.block_layer(num_blocks=num_blocks, in_channel=layer2_channel, out_channel=out_channel)
        # END 

    def block_layer(self, num_blocks, in_channel, out_channel):
        """
        Inputs:
        num_blocks = number of blocks in the block layer
        in_channel = number of input channels to the entire block layer
        out_channel = number of output channels in the output of the entire block layer
        """
        # make use of the ResidualBlock class you've already implemented
        # again, note interm_channel == out_channel for all residual blocks here
        # TODO: implement the block layer which has num_blocks blocks stacked together

        layers = []
        layers.append(
          ResidualBlock(
            in_channel=in_channel,
            interm_channel=out_channel,
            out_channel=out_channel
          )
        )  

        for _ in range(1, num_blocks):
          layers.append(
            ResidualBlock(
              in_channel=out_channel,
              interm_channel=out_channel,
              out_channel=out_channel
            )
          ) 

        return nn.Sequential(*layers)     
        # END TODO
        

    def forward(self, x):
        # TODO: implement the forward function based on the architecture described above
        x = self.last(self.layer2(self.layer1(self.first(x))))
        return x
        # END TODO
