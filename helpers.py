import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, RandomHorizontalFlip, Grayscale, Normalize
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score

def create_dataset(root, transformation):
    dataset = ImageFolder(root, transformation)
    return dataset

def produce_loader(data, batch_size, sampler=None, shuffle=False):
    loader = torch.utils.data.DataLoader(data, batch_size = batch_size, sampler=sampler, shuffle=shuffle)
    return loader

def visualize_data(dataset, figsize=(8,8), axes=3):
    indices = []
    labels_map = {
        0: "normal",
        1: "pneumonia",
    }
    cols, rows = axes, axes
    figure = plt.figure(figsize=figsize)
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        indices.append(sample_idx)
        img, label = dataset[sample_idx]
        if not (type(dataset)==list):
            img = img.swapaxes(0,1)
            img = img.swapaxes(1,2)
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img)
    print(indices)
    indices = []
    plt.show()

def test(device, model, data_loader, criterion=nn.CrossEntropyLoss(), autoencoder=None, get_predictions=False):
    # Use cross-entropy loss function
    model.eval()
    # Initialize epoch loss and accuracy
    test_loss = 0.0
    correct = 0
    total = 0
    # Get list of predictions for confusion matrix
    if get_predictions:
        true_labels = torch.tensor([]).to(device)
        model_preds = torch.tensor([]).to(device)
    # Iterate over test data
    for inputs, labels in data_loader:
        # Get from dataloader and send to device
        inputs = inputs.to(device)
        if autoencoder:
            inputs = autoencoder.get_features(inputs)
        labels = labels.to(device)
        # Compute model output and loss
        # (No grad computation here, as it is the test data)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
        if get_predictions:
            true_labels = torch.cat((true_labels, labels))
            model_preds = torch.cat((model_preds, predicted)) 
        # Accumulate loss and correct predictions for epoch
        test_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # Calculate epoch loss and accuracy
    test_loss /= len(data_loader)
    test_acc = correct/total
    if get_predictions:
        true_labels = true_labels.type(torch.int64).cpu().numpy()
        model_preds = model_preds.type(torch.int64).cpu().numpy()
        print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')
        return true_labels, model_preds, test_loss, test_acc
    return test_loss, test_acc

def train(device, model, train_loader, valid_loader, optimizer, epochs, criterion=nn.CrossEntropyLoss(), autoencoder=None):
    # Performance curves data
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        # Initialize epoch loss and accuracy
        epoch_loss = 0.0
        correct = 0
        total = 0
        # Iterate over training data
        for batch_number, (inputs, labels) in enumerate(train_loader):
            # Get from dataloader and send to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Zero out gradients
            optimizer.zero_grad()
            # Compute model output and loss
            if autoencoder:
                inputs = autoencoder.get_features(inputs)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            # Backpropagate loss and update model weights
            loss.backward()
            optimizer.step()
            # Accumulate loss and correct predictions for epoch
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
            if (batch_number%5==0):
                print(f'Epoch {epoch+1}/{epochs}, Batch number: {batch_number}, Cumulated accuracy: {correct/total}')
        # Calculate epoch loss and accuracy
        epoch_loss /= len(train_loader)
        epoch_acc = correct/total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f'--- Epoch {epoch+1}/{epochs}: Train loss: {epoch_loss:.4f}, Train accuracy: {epoch_acc:.4f}')
        
        # Validation set
        val_loss, val_acc = test(device, model, valid_loader, criterion, autoencoder)
        if val_acc >= 0.90:
            torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 
            f'./{model.__class__.__name__}_{epoch+1}epochs')
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f'--- Epoch {epoch+1}/{epochs}: Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}')      
    return train_losses, train_accuracies, val_losses, val_accuracies

def show_metrics(true_labels, model_preds):
    cm = confusion_matrix(true_labels, model_preds)
    ConfusionMatrixDisplay(cm, display_labels=['normal', 'pneumonia']).plot()
    print(f'Precision: {precision_score(true_labels, model_preds)}')
    print(f'Recall: {recall_score(true_labels, model_preds)}')
    print(f'F1 score: {f1_score(true_labels, model_preds)}')

def return_metrics(true_labels,model_preds):
    cm = confusion_matrix(true_labels, model_preds)
    p=precision_score(true_labels, model_preds)
    r=recall_score(true_labels, model_preds)
    f=f1_score(true_labels, model_preds)
    return p, r, f
    
    
def get_pictures_test(device, model, data_loader, autoencoder=None):
    # Use cross-entropy loss function
    model.eval()
    # Initialize epoch loss and accuracy
    correct_list = []
    wrong_list = []
    true_labels = torch.tensor([]).to(device)
    model_preds = torch.tensor([]).to(device)
    # Iterate over test data
    for inputs, labels in data_loader:
        # Get from dataloader and send to device
        inputs_ = inputs.to(device)
        if autoencoder:
            inputs_ = autoencoder.get_features(inputs_)
        labels = labels.to(device)
        # Compute model output and loss
        # (No grad computation here, as it is the test data)
        with torch.no_grad():
            outputs = model(inputs_)
            _, predicted = torch.max(outputs.data, 1)
            true_labels = torch.cat((true_labels, labels))
            model_preds = torch.cat((model_preds, predicted)) 
            if (predicted.item() == labels.item()):
                correct_list.append( (inputs.squeeze().detach().cpu().numpy(), labels.item()) )
            else:
                wrong_list.append( (inputs.squeeze().detach().cpu().numpy(), labels.item()) )
    true_labels = true_labels.type(torch.int64).cpu().numpy()
    model_preds = model_preds.type(torch.int64).cpu().numpy()
    return correct_list, wrong_list, true_labels, model_preds