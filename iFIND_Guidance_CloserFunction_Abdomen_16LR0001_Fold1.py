from datasets import guidancedatasetnew as gdataset
from torchvision import transforms as torchtransforms
from models.regressors import ConvNet, Regressor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# In this script instead of using the proximity as the ground-truth, I take the original labels from Emily, i.e. only
# saying which are standard plane and which aren't. Then I will input at least two sequence of images, and compared to
# the labels (so to the frame containing the standard view) it will tell the user whether it is getting 'closer'
# or 'further' from the standard view.
params = {}
params['BATCH_SIZE'] = 64
params['MAX_EPOCHS'] = 100
params['LR'] = 0.000001
params['do_augmentation'] = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if params['do_augmentation'] == True:
    transform = torchtransforms.Compose([torchtransforms.ToPILImage(),
                                     torchtransforms.RandomHorizontalFlip(),
                                     torchtransforms.RandomAffine(degrees=10.0, translate=(0.25, 0.25), scale=(0.7, 1.3)),
                                     torchtransforms.ToTensor()
                                     ])
else:
    transform = torchtransforms.Compose([torchtransforms.ToTensor()])

validation_transforms = torchtransforms.Compose([torchtransforms.ToTensor()])


trainset = gdataset.guidancedatasetnew(datafile = r'C:\Users\ng20\Desktop\Guidance_Work\Nooshin_find2\Python_Codes\New_Fresh_Code\CNN_9Patients\training.txt', transform= transform, preload = True)
validationset = gdataset.guidancedatasetnew(datafile = r'C:\Users\ng20\Desktop\Guidance_Work\Nooshin_find2\Python_Codes\New_Fresh_Code\CNN_9Patients\validation.txt', transform= transform, preload = True)

sample = trainset[0]
trainloader = torch.utils.data.DataLoader(trainset, batch_size=params['BATCH_SIZE'], shuffle=False)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=params['BATCH_SIZE'], shuffle=False)


model = Regressor(sample[0].shape[1:]).to(device)
model = model.double()
#criterion = nn.SmoothL1Loss() # a.k.a HUBER loss: criterion that uses a squared term if the absolute element-wise error falls below 1 and an L1 term otherwise. It is less sensitive than the previous MSE loss for large values
#criterion = nn.L1Loss()
criterion = nn.CrossEntropyLoss() # need to use the cross-entropy loss because now it is more a classification problem rather than regression
optimizer = optim.Adam(model.parameters(), lr=params['LR'],weight_decay=1e-3) # optimiser, the weight decay adds a a regularisation term to the optimiser

losses = []
validation_losses = []
for epoch in range(params['MAX_EPOCHS']):
    running_loss = 0
    for i, (inputs, labels) in enumerate(trainloader):

        inputs = inputs.to(device).double()
        #inputs=inputs.double()
        outputs = model(inputs)  # passes a batch of images to the model in the forward pass
        labels=labels.type(torch.LongTensor)
        labels = labels.to(device)
        loss = criterion(outputs.float(), labels)  # computes the mean squared error between outputs and true labels
        running_loss += loss.item()  # each time the loss is appended to a list so that at the end it can be plotted

        # Backpropagation and perform Adam optimisation
        optimizer.zero_grad()  # start by setting the gradients to zero
        loss.backward()  # this step performs the backpropagation on the calculated loss to find the gradients
        optimizer.step()  # Adam optimiser is called
    running_loss /= len(trainloader.dataset)
    losses.append(running_loss)

    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        running_validation_loss = 0
        for i, (inputs, labels) in enumerate(validationloader):
            inputs = inputs.to(device).double()
            # inputs=inputs.double()
            outputs = model(inputs)  # passes a batch of images to the model in the forward pass
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            loss = criterion(outputs, labels)  # computes the mean squared error between outputs and true labels
            running_validation_loss += loss.item()  # each time the loss is appended to a list so that at the end it can be plotted
            _, predicted = torch.max(outputs.data, 1) # find the maximum value from the output
            total += labels.size(0) # total image/label size
            correct += (predicted == labels).sum().item() # compute how many correct predictions there are

        running_validation_loss /= len(validationloader.dataset)
        validation_losses.append(running_validation_loss)
        print('[{}] T: {:.6f} | V: {:.6f}'.format(epoch, running_loss, running_validation_loss))


    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


plt.plot(losses)
plt.plot(validation_losses)
plt.xlabel('epochs')
plt.title('Loss')
plt.legend('Training loss','Validation loss')
plt.show()


