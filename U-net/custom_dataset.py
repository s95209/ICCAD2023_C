import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_folders = [folder for folder in os.listdir(root_dir) if folder.startswith("testcase")]
        
        # # Initialize dictionaries to store max and min values for each input channel
        # self.max_values = {"current_data": 0, "eff_dist_data": 0, "pdn_density_data": 0}
        # self.min_values = {"current_data": 0, "eff_dist_data": 0, "pdn_density_data": 0}
        
        # # Find global max and min for each input channel
        # for folder_name in self.data_folders:
        #     folder_number = folder_name.split("testcase")[1]
        #     current_data_path = os.path.join(self.root_dir, folder_name, f"current_map{folder_number}_current.csv")
        #     eff_dist_data_path = os.path.join(self.root_dir, folder_name, f"current_map{folder_number}_eff_dist.csv")
        #     pdn_density_data_path = os.path.join(self.root_dir, folder_name, f"current_map{folder_number}_pdn_density.csv")

        #     current_data = np.genfromtxt(current_data_path, delimiter=',')
        #     eff_dist_data = np.genfromtxt(eff_dist_data_path, delimiter=',')
        #     pdn_density_data = np.genfromtxt(pdn_density_data_path, delimiter=',')

        #     self.max_values["current_data"] = max(self.max_values["current_data"], current_data.max())
        #     self.min_values["current_data"] = min(self.min_values["current_data"], current_data.min())

        #     self.max_values["eff_dist_data"] = max(self.max_values["eff_dist_data"], eff_dist_data.max())
        #     self.min_values["eff_dist_data"] = min(self.min_values["eff_dist_data"], eff_dist_data.min())

        #     self.max_values["pdn_density_data"] = max(self.max_values["pdn_density_data"], pdn_density_data.max())
        #     self.min_values["pdn_density_data"] = min(self.min_values["pdn_density_data"], pdn_density_data.min())
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(self.max_values)
        # print(self.min_values)
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


    def __len__(self):
        return len(self.data_folders)

    def __getitem__(self, index):
        folder_name = self.data_folders[index]
        folder_number = folder_name.split("testcase")[1]
        current_data_path = os.path.join(self.root_dir , folder_name, f"current_map{folder_number}_current.csv")
        eff_dist_data_path = os.path.join(self.root_dir , folder_name, f"current_map{folder_number}_eff_dist.csv")
        pdn_density_data_path = os.path.join(self.root_dir , folder_name, f"current_map{folder_number}_pdn_density.csv")
        ir_drop_data_path = os.path.join(self.root_dir , folder_name, f"current_map{folder_number}_ir_drop.csv")

        current_data = np.genfromtxt(current_data_path, delimiter=',')
        eff_dist_data = np.genfromtxt(eff_dist_data_path, delimiter=',')
        pdn_density_data = np.genfromtxt(pdn_density_data_path, delimiter=',')
        ir_drop_data = np.genfromtxt(ir_drop_data_path, delimiter=',')

        # Normalize each input channel using global max and min values
        # current_data = (current_data - self.min_values["current_data"]) / (self.max_values["current_data"] - self.min_values["current_data"])
        # eff_dist_data = (eff_dist_data - self.min_values["eff_dist_data"]) / (self.max_values["eff_dist_data"] - self.min_values["eff_dist_data"])
        # if self.max_values["pdn_density_data"] != self.min_values["pdn_density_data"]:
        #     pdn_density_data = (pdn_density_data - self.min_values["pdn_density_data"]) / (self.max_values["pdn_density_data"] - self.min_values["pdn_density_data"])
        # else:
        #     pdn_density_data = pdn_density_data - pdn_density_data  # Avoid division by zero
            
        current_data = (current_data - 0) / (3.74722 * (10 ** -7) - 0)
        eff_dist_data = (eff_dist_data - 0) / (61.3528 - 0)
        pdn_density_data = (pdn_density_data - 0) / (3.0 - 0)
        # print(current_data)
        # Combine the three input channels into a single tensor
        input_data = np.stack((current_data, eff_dist_data, pdn_density_data), axis=0)
        # Convert to PyTorch tensor
        input_data = torch.FloatTensor(input_data)
        # Convert ground truth data to PyTorch tensor
        ir_drop_data = torch.FloatTensor(ir_drop_data * 100000).unsqueeze(0)
        return input_data, ir_drop_data
