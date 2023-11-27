import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt

from base_cnn import BaseCNN


def validate_model(model: BaseCNN, validation_loader: DataLoader):
    model.eval()
    validation_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(model.device), target.to(model.device)
            output = model(data)
            validation_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(validation_loader.dataset)
    accuracy = correct / len(validation_loader.dataset)

    return validation_loss, accuracy


def train(model: BaseCNN,
          train_loader: DataLoader,
          val_loader: DataLoader,
          patience=2,
          check_train=False
          ):
    model.to(model.device)
    train_losses = []
    train_counter = []

    optimizer = model.make_optimizer()

    # For early stopping
    best_accuracy = 0.0
    earlystop_count = 0

    for epoch in range(model.epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            inputs = model.do_additional_transforms(inputs)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = model.criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
                # Store train loss
                train_losses.append(running_loss / 100)
                train_counter.append(
                    epoch + (i / len(train_loader))
                )
                running_loss = 0.0

        # Validation
        validation_loss, validation_accuracy = validate_model(model, val_loader)
        print(f'Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy * 100:.2f}%')

        if check_train:
            train_loss, train_accuracy = validate_model(model, train_loader)
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%')

        # Check Early Stopping
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            earlystop_count = 0
        else:
            earlystop_count += 1

        if earlystop_count > patience:
            print(f"Early stopping! No improvement for the last {patience} epochs.")
            break

    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.xlabel('Epoch (values between ints indicate batches)')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs. Epoch')
    plt.show()


def test(model: BaseCNN, test_loader: DataLoader):
    model.eval()
    test_loss = 0
    # correct = 0

    # For class count predictions
    correct_pred = {classname: 0 for classname in model.classes}
    total_pred = {classname: 0 for classname in model.classes}

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(model.device), targets.to(model.device)
            output = model(data)
            test_loss += F.nll_loss(output, targets, reduction='sum').item()  # Calculate the test loss
            preds = output.argmax(dim=1, keepdim=True)  # Get the index of the maximum log-probability
            for pred, label in zip(preds, targets):
                correct_pred[model.classes[label]] += pred.eq(label.view_as(pred)).item()  # Count correct predictions
                total_pred[model.classes[label]] += 1

    # print accuracy for each class
    correct = 0
    for classname, correct_count in correct_pred.items():
        correct += correct_count
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {correct_count}/{total_pred[classname]} ({accuracy:.2f}%)')

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
