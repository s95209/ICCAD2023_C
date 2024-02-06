import os
import numpy as np
import torch
from custom_dataset import CustomDataset
from model import Autoencoder

torch.set_printoptions(profile="full")

def calculate_f1_score(predicted_data, target_data):
    # Calculate threshold as 90% of the maximum value in target_data
    threshold = np.max(target_data)*0.9

    # Binarize predicted_data and target_data based on the threshold
    predicted_data_binary = (predicted_data > threshold).astype(int)
    target_data_binary = (target_data > threshold).astype(int)
    
    TP = np.sum((target_data_binary == 1) & (predicted_data_binary == 1))
    FP = np.sum((target_data_binary == 0) & (predicted_data_binary == 1))
    FN = np.sum((target_data_binary == 1) & (predicted_data_binary == 0))
    
    # print(TP, " ", FP, " ", FN)

    if TP + FP == 0:
        precision = 0.0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    return f1_score

def test_model(data_folder_path, encoder_weights_path, decoder_weights_path, output_folder_path):
    # Create an instance of the Autoencoder model
    model = Autoencoder(encoder_weights_path, decoder_weights_path)

    if torch.cuda.is_available():
        print("cuda_is_available")
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Create a custom dataset
    dataset = CustomDataset(data_folder_path)

    # Loop through the dataset and make predictions
    f1notzero = 0
    total_f1_score = 0
    for i in range(len(dataset)):
        inputs, targets = dataset[i]
        inputs = inputs.unsqueeze(0)  # Add a batch dimension
        targets = targets.unsqueeze(0)

        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
        outputs = outputs / 100000
        targets = targets / 100000
        # Calculate Mean Absolute Error (MAE) Loss
        mae_loss = torch.nn.functional.l1_loss(outputs , targets)
        print(f"Image {dataset.data_folders[i]}, MAE Loss: {mae_loss.item()}")
        
        
        # Convert tensors to numpy arrays
        targets_np = targets.squeeze().cpu().numpy()
        outputs_np = outputs.squeeze().cpu().numpy()

        targets_f1 = targets.squeeze().detach().cpu().numpy()
        outputs_f1 = outputs.squeeze().detach().cpu().numpy()

        f1_score = calculate_f1_score(outputs_f1, targets_f1)
        total_f1_score += f1_score

        if(f1_score != 0):
            f1notzero += 1
            
        print("F1 score:", f1_score)
        # print(targets - outputs)
        print("================================================")
        # Get the folder name of the current testcase
        folder_name = dataset.data_folders[i]
        folder_number = folder_name.split("testcase")[1]

        # Create output folder with the same name as the input testcase
        output_case_folder = os.path.join(output_folder_path, f"testcase{folder_number}")
        os.makedirs(output_case_folder, exist_ok=True)

        # Save the inputs, targets, and outputs as CSV files in the testcase folder
        target_filename = os.path.join(output_case_folder, f"target_image.csv")
        output_filename = os.path.join(output_case_folder, f"output_image.csv")

        np.savetxt(target_filename, targets_np, delimiter=",")
        np.savetxt(output_filename, outputs_np, delimiter=",")

        log_filename = os.path.join(output_case_folder, f"loss.txt")
        log_file = open(log_filename, "w")
        log_file.write(f" MAE Loss: {mae_loss.item()}\n")
        log_file.close()
        
        log_filename2 = os.path.join(output_case_folder, f"F1.txt")
        log_file2 = open(log_filename2, "w")
        log_file2.write(f" F1: {f1_score}\n")
        log_file2.close()
    
    return f1notzero, total_f1_score

if __name__ == "__main__":
    # data_folder_path = '/home/s111062697/ICCAD2023/circuit_data/fake-circuit-data_20230623'
    data_folder_path = '/home/s111062697/ICCAD2023/circuit_data/real-circuit-data_20230615'
    encoder_weights_path = '/home/s111062697/ICCAD2023/U-net/weight/encoder_weights_best.pth'
    decoder_weights_path = '/home/s111062697/ICCAD2023/U-net/weight/decoder_weights_best.pth'
    output_folder_path = '/home/s111062697/ICCAD2023/U-net/output'

    test_model(data_folder_path, encoder_weights_path, decoder_weights_path, output_folder_path)
