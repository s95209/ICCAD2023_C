import pandas as pd
import matplotlib.pyplot as plt
import os


def process_testcases_in_directory(base_dir):
    # Get a list of all subdirectories (testcase folders) in the base directory
    testcase_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("testcase")]
    # Loop through each testcase folder and process the CSV files
    for testcase_dir in testcase_dirs:
        output_csv_path = os.path.join(base_dir, testcase_dir, "output_image.csv")
        target_csv_path = os.path.join(base_dir, testcase_dir, "target_image.csv")

        output_png_path = os.path.join(base_dir, testcase_dir, "output_image.png")
        target_png_path = os.path.join(base_dir, testcase_dir, "target_image.png")
        
        # Read the CSV files into pandas DataFrames
        output_df = pd.read_csv(output_csv_path)
        target_df = pd.read_csv(target_csv_path)
        
        # Assuming your CSV data is arranged in a way that represents the heatmap
        # (if not, you may need to preprocess the data accordingly)
        
        # Convert DataFrames to numpy arrays for plotting
        output_data = output_df.to_numpy()
        target_data = target_df.to_numpy()
        
        vmin = min(output_data.min(), target_data.min())
        vmax = max(output_data.max(), target_data.max())
        
        # Create subplots for both output and target heatmaps
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        
        # Plot the output heatmap
        output_heatmap = axes[0].imshow(output_data, interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[0].set_title('Output Heatmap')
        plt.colorbar(output_heatmap, ax=axes[0])  # Add colorbar
        
        # Plot the target heatmap
        target_heatmap = axes[1].imshow(target_data, interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[1].set_title('Target Heatmap')
        plt.colorbar(target_heatmap, ax=axes[1])  # Add colorbar
        
        # Save the figure as PNG
        plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free up memory


if __name__ == "__main__":
    base_dir = "/home/s111062697/ICCAD2023/U-net/output"  # Replace with the actual base directory
    process_testcases_in_directory(base_dir)
