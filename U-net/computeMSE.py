import argparse
import numpy as np
import pandas as pd

def calculate_mse(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    image1 = df1.to_numpy()
    image2 = df2.to_numpy()
    
    if image1.shape != image2.shape:
        raise ValueError("圖片維度不一致，無法計算MSE。")
    else:
        print(image1.shape ," ",image2.shape)
    mse = np.mean((image1 - image2) ** 2)
    return mse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Mean Squared Error (MSE) between two CSV files.")
    parser.add_argument("file1", help="first filename")
    parser.add_argument("file2", help="second filename")
    args = parser.parse_args()
    
    try:
        mse_result = calculate_mse(args.file1, args.file2)
        print("mse: ", mse_result)
    except FileNotFoundError as e:
        print(f"找不到檔案: {e.filename}")
    except ValueError as e:
        print(str(e))