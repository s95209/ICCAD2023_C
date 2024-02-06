import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from model import Autoencoder
import math
import test as F1test
    
def create_validation_dataloader(data_folder_path):
    dataset = CustomDataset(data_folder_path)
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def create_training_dataloader(data_folder_path):
    dataset = CustomDataset(data_folder_path)
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_model(data_folder_path, encoder_weights_path, decoder_weights_path):
    # Create an instance of the Autoencoder model
    
    if os.path.exists(encoder_weights_path) and os.path.exists(decoder_weights_path):
        model = Autoencoder(encoder_weights_path, decoder_weights_path)
    else:
        model = Autoencoder()

    if torch.cuda.is_available():
        print("cuda_is_available")
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

    model.to(device)
    
    # Define the loss function and optimizer
    # criterion_MSE = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dataloader
    training_data_folder_path = '/home/s111062697/ICCAD2023/circuit_data/fake-circuit-data_20230623'
    train_dataloader = create_training_dataloader(training_data_folder_path) #training dataloader
    
    validation_data_folder_path = '/home/s111062697/ICCAD2023/circuit_data/real-circuit-data_20230615'
    val_dataloader = create_validation_dataloader(validation_data_folder_path) #validation dataloader

    best_val_loss = float('inf')  # Initialize with a large value to track the best validation loss

    # Training loop
    n_epochs, best_loss = 1000, math.inf
    log_file = open("training_log.txt", "w")
    bestf1notzero = 10
    bestf1total = 0
    
    for epoch in range(n_epochs):
        print(f"==============Epoch {epoch+1}/{n_epochs}==============")
        running_loss = 0.0
        
        # Training phase
        model.train()
        for inputs, targets in train_dataloader:
            optimizer.zero_grad() # Zero the gradients
            inputs, targets = inputs.to(device), targets.to(device) # data move to device
            # Forward pass
            pred = model(inputs)    
            loss = criterion(pred, targets)
            loss.backward() # Compute gradient(backpropagation)
            optimizer.step() # Update parameters

            # Update running loss
            running_loss += loss.item()
        # Print average loss for this epoch
        mean_train_loss = running_loss / len(train_dataloader)
        log_file.write(f"Epoch {epoch+1}/{n_epochs}")
        log_file.write (f"Loss/train: {mean_train_loss}\n")
        print(f"Epoch {epoch+1}/{n_epochs}, Loss/train: {mean_train_loss}\n")
        
        # #evaluation phase
        # running_loss = 0.0
        # model.eval() # Set your model to evaluation mode.
        # for inputs, targets in val_dataloader:
        #     inputs, targets = inputs.to(device), targets.to(device)
        #     with torch.no_grad():
        #         pred = model(inputs)
        #         loss = criterion(pred, targets)
        #         running_loss += loss.item()
        # mean_val_loss = running_loss / len(val_dataloader)
        # log_file.write(f"Loss/val: {mean_val_loss}\n")
        # print(f"Epoch {epoch+1}/{n_epochs}, Loss/val: {mean_val_loss}\n")
        d_folder_path = '/home/s111062697/ICCAD2023/circuit_data/real-circuit-data_20230615'
        en_weights_path = '/home/s111062697/ICCAD2023/U-net/weight/encoder_weights.pth'
        de_weights_path = '/home/s111062697/ICCAD2023/U-net/weight/decoder_weights.pth'
        output_folder_path = '/home/s111062697/ICCAD2023/U-net/output'
        torch.save(model.encoder.state_dict(), encoder_weights_path)
        torch.save(model.decoder.state_dict(), decoder_weights_path)   
        f1_notzero, f1_total = F1test.test_model(d_folder_path, en_weights_path, de_weights_path, output_folder_path)

        if bestf1notzero < f1_notzero:
            # best_loss = mean_train_loss
            print("save best weight")
            bestf1notzero = f1_notzero
            bestf1total = f1_total
            torch.save(model.encoder.state_dict(), '/home/s111062697/ICCAD2023/U-net/weight/encoder_weights_best.pth')
            torch.save(model.decoder.state_dict(), '/home/s111062697/ICCAD2023/U-net/weight/decoder_weights_best.pth')
        elif bestf1notzero == f1_notzero:
            if bestf1total <= f1_total:
                print("save best weight")
                bestf1notzero = f1_notzero
                bestf1total = f1_total
                torch.save(model.encoder.state_dict(), '/home/s111062697/ICCAD2023/U-net/weight/encoder_weights_best.pth')
                torch.save(model.decoder.state_dict(), '/home/s111062697/ICCAD2023/U-net/weight/decoder_weights_best.pth')

        print("===========================================================")     
    log_file.close()

    # Save the trained encoder and decoder weights


if __name__ == "__main__":
    data_folder_path = '/home/s111062697/ICCAD2023/circuit_data/fake-circuit-data_20230623'
    encoder_weights_path = '/home/s111062697/ICCAD2023/U-net/weight/encoder_weights.pth'
    decoder_weights_path = '/home/s111062697/ICCAD2023/U-net/weight/decoder_weights.pth'
    train_model(data_folder_path, encoder_weights_path, decoder_weights_path)