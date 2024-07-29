import sys

# Ignore IPython kernel arguments
sys.argv = sys.argv[:1]

import copy
from time import time
import matplotlib.pyplot as plt

import torch

import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from arguments import parse_args
from ubi_dataset import load_dataset
from meter import RunningMeter, BestMeter
from model import Classifier
from utils import save_meter, save_model, \
    compute_best_metrics, \
    compute_classifier_metrics, set_all_seeds,\
    load_classifier_model, compute_classifier_metrics_final


# --------------------------------------k∆ˆ----------------------------------------


def evaluate_with_classifier(args):
    """
    Evaluating the performance of CPC with a MLP classifier
    :param args: arguments
    :return: None
    """

    # Loading the data
    data_loaders, dataset_sizes = load_dataset(args, classifier=True)

    # Creating the model
    model = Classifier(args).to(args.device)

    # Loading pre-trained weights if available
    if args.saved_model is not None:
        model.load_pretrained_weights(args)

    # Optimizer settings
    
    #optimizer = optim.Adam(model.parameters(), lr=args.classifier_lr)
    #scheduler = StepLR(optimizer, step_size=25, gamma=0.8)
    optimizer = optim.Adadelta(model.parameters(), lr=args.classifier_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    if args.dataset=="MIMIC":
        class_weights_tensor = torch.FloatTensor([1.16, 7.39]).to(args.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    if args.dataset=="MESA":
        class_weights_tensor = torch.FloatTensor([2.37, 1.73]).to(args.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    if args.dataset=="GLOBEM":
        class_weights_tensor = torch.FloatTensor([0.89, 2.12]).to(args.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
      
    # Tracking meters
    running_meter = RunningMeter(args=args)
    best_meter = BestMeter()
    train_losses, val_losses = [], []

    for epoch in range(0, args.num_epochs):
        since = time()
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        # Training
        model, optimizer, scheduler = train(model, data_loaders["train"],
                                            criterion, optimizer, scheduler,
                                            args, epoch,
                                            dataset_sizes["train"],
                                            running_meter)
        
    
        train_loss = running_meter.loss["train"][-1]
        train_losses.append(train_loss)

        # Validation
        evaluate(model, data_loaders["val"], args, criterion, epoch,
                 phase="val", dataset_size=dataset_sizes["val"],
                 running_meter=running_meter)    
         
        val_loss = running_meter.loss["val"][-1]
        val_losses.append(val_loss)

        # Evaluating on the test data
        evaluate(model, data_loaders["test"], args, criterion, epoch,
                 phase="test", dataset_size=dataset_sizes["test"],
                 running_meter=running_meter)

        # Saving the logs
        save_meter(args, running_meter, finetune=True)

         # Updating the best weights
        if running_meter.loss["val"][-1] < best_meter.loss["val"]:
            print('Updating the best val loss at epoch: {}, since {} < '
                  '{}'.format(epoch, running_meter.loss["val"][-1],
                              best_meter.loss["val"]))
            best_meter = compute_best_metrics(running_meter, best_meter)
            running_meter.update_best_meter(best_meter)
            save_meter(args, running_meter)

            best_model_wts = copy.deepcopy(model.state_dict())

        # Printing the time taken
        time_elapsed = time() - since
        print('Epoch {} completed in {:.0f}m {:.0f}s'
              .format(epoch, time_elapsed // 60, time_elapsed % 60))

    
    # Printing the best metrics corresponding to the highest validation
    #loss
    best_meter.display()

    model.load_state_dict(best_model_wts)

    print('Saving the trained model!')
    save_model(model, args, classifier=True)

    # Plotting the training and validation loss
    plt.plot(range(args.num_epochs), train_losses, label='Training Loss')
    plt.plot(range(args.num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('classifier_loss.png')
    plt.savefig('cpc_classifier_loss.png')
    plt.show()

    return



def train(model, data_loader, criterion, optimizer,
           scheduler, 
           args, epoch,
          dataset_size, running_meter):
    # Setting the model to training mode
    model.train()

    # Freeze encoder layers
    if args.learning_schedule == 'last_layer':
        model.freeze_encoder_layers()

    # To track the loss and other metrics
    running_loss = 0.0
    actual_labels = []
    pred_labels = []

    # Iterating over the data
    for inputs, labels in data_loader:
        inputs = inputs.float().to(args.device)
        labels = labels.long().to(args.device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)
        actual_labels.extend(labels.cpu().data.numpy())
        pred_labels.extend(preds.cpu().data.numpy())

    scheduler.step()

    # Statistics
    loss = running_loss / dataset_size
    _ = compute_classifier_metrics(actual_labels, pred_labels,
                                   'train', running_meter, loss,
                                   epoch)

    return model, optimizer, scheduler
    #return model, optimizer


def evaluate(model, data_loader, args, criterion, epoch, phase, dataset_size,
             running_meter):
    # Setting the model to eval mode
    model.eval()

    # To track the loss and other metrics
    running_loss = 0.0
    actual_labels = []
    pred_labels = []

    # Iterating over the data
    for inputs, labels in data_loader:
        inputs = inputs.float().to(args.device)
        labels = labels.long().to(args.device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)
        actual_labels.extend(labels.cpu().data.numpy())
        pred_labels.extend(preds.cpu().data.numpy())

    # Statistics
    loss = running_loss / dataset_size
    _ = compute_classifier_metrics(actual_labels, pred_labels,
                                   phase, running_meter, loss,
                                   epoch)

    return


def get_metrics(model, data_loader, args):
    # Setting the model to eval mode
    model.eval()

    # To track the loss and other metrics
    running_loss = 0.0
    actual_labels = []
    pred_labels = []
    probs = []

    # Iterating over the data
    for inputs, labels in data_loader:
        inputs = inputs.float().to(args.device)
        labels = labels.long().to(args.device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            probabilities = nn.functional.softmax(outputs, dim=1)
            probabilities = probabilities[:, 1]
            _, preds = torch.max(outputs, 1)

        # Appending predictions and loss
        actual_labels.extend(labels.cpu().data.numpy())
        pred_labels.extend(preds.cpu().data.numpy())
        probs.extend(probabilities.cpu().data.numpy())

    _ = compute_classifier_metrics_final(actual_labels, pred_labels,probs, args)

    return pred_labels, probs




# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    set_all_seeds(args)
    print(args)

    #evaluate_with_classifier(args=args)

    print('------ Evaluation complete! ------')


    model = load_classifier_model(args=args)
    data_loaders, dataset_sizes = load_dataset(args, classifier=True)
    preds, probs = get_metrics(model, data_loaders["test"], args)
