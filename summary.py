import numpy as np
import os 
import argparse

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description= "Report")
    parser.add_argument("--result_folder", type=str, default="log/idealW", help="Path to folder containing .npy files")

    args = parser.parse_args() 

    # initialize
    success_rate = 0 
    l2_distance = 0
    count = 0 

    # loop
    for dirpath, dirname, filenames in os.walk(args.result_folder): 
        for filename in filenames: 
            data = np.load(os.path.join(dirpath, filename), allow_pickle=True).item()
            success_rate += data['success']
            l2_distance += np.sqrt(data['l2']) 
            count += 1

    print(f"Success rate: {round(success_rate*100/count, 4)} %") 
    print(f"L2 distance: {round(l2_distance / count, 4)}")

