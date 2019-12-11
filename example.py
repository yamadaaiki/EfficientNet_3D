from efficientnet_pytorch import EfficientNet
import numpy as np
import torch

model = EfficientNet.from_name("efficientnet-b0", override_params={'num_classes': 2})

from torchsummary import summary
summary(model, input_size=(1, 200, 1024, 200))

model = model.to("cuda:3")
inputs = torch.randn((1, 1, 200, 1024, 200)).to("cuda:3")
labels = torch.tensor([0]).to("cuda:3")
# test forward
num_classes = 2

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()
for epoch in range(2):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    print('[%d] loss: %.3f' % (epoch + 1, loss.item()))

print('Finished Training')
