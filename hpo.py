#Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import argparse
import json
import logging
import os
import sys


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

def test(model, test_loader,criterion):
    '''
    Complete this function that can take a model and a
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval() #Eval mode
    test_loss = 0
    correct = 0
    with torch.no_grad() :
        for data,target in test_loader:
            output = model(data) #predict
            test_loss += criterion(output,target).item() #sum up batch loss
            pred = output.max(1, keepdim=True)[1] #get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() #Geting correct predictions according to target
    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer):
    '''
    Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    #Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            optimizer.zero_grad() #Resets gradients for new batch
            output = model(data) #Runs forward pass
            loss = criterion(output, target) #Calculates loss
            loss.backward() #Calculates Gradients for Model Parameters
            optimizer.step() #Updates weigths
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    return model
    
def net():
    '''
    Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False #Freezing internal layers of the pre-trainned model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,10) #Setting the first fully connected layer that will be trainned
    return model

def create_data_loaders(dir,batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("Get test data loader")
    if train_mode:
    #Setting transforms in training/eval data
    transform = transforms.Compose([transforms.Resize((400, 400)),
                       transforms.RandomRotation(10),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])
    #Grouping data from classes folders and apply transformers
    data = datasets.ImageFolder(root = dir, transform = transform)
    #Spliting into train/eval datasets
    total_len = len(data)
    training_len = int(0.8 * total_len)
    eval_len = total_len - training_len
    training_data, eval_data = torch.utils.data.random_split(data, (training_len, eval_len))
    #Creating loaders
    train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_data,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader,eval_loader

def save_model(model,model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

def main(args):
    '''
    Initialize a model by calling the net function
    '''
    model=net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}.")
    model = model.to(device)
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    # Setting data loaders
    train_loader = create_data_loaders(args.data_dir, ags.batch_size)[0]
    eval_loader = create_data_loaders(args.data_dir, ags.test-batch-size)[1]
    #Training loop
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, eval_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    save_model(model,args.model_dir)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args=parser.parse_args()
    
    main(args)
