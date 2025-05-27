from torchvision import transforms
from src.model import SignNN 
from src.loss_function import UnTargeted_idealW, UnTargeted_realW
from src.attack import Attack_idealW, Attack_realW
import argparse
import os
from src.utils import *
from PIL import Image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="idealW", help="Attack on ideal world or real world")
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--mut", type=float, default=0.3)
    parser.add_argument("--s", type=int, default=20)
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--li", type=int, default=4)
    parser.add_argument("--data_dicrectory", type=str, default="data/TEST", help="Image File directory")
    parser.add_argument("--save_directory", type=str,default="log/idealW", help="Where to store the .npy files with the results")
    args = parser.parse_args()
    mode = args.mode
    saved_folder = args.save_directory
    os.makedirs(saved_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use {device}")
    model = SignNN().to(device)
    model.eval() 
    model.load_state_dict(torch.load('saved_model/best_f1.pt'))

    X = [] 
    y = [] 
    for label in range(9): 
        folder_path = os.path.join(args.data_dicrectory, str(label))
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                img = transform(Image.open(img_path))
                X.append(img)
                y.append(label)
    print("Số lượng ảnh:", len(X))
    print("Shape ảnh:", X[0].shape)
    print("Nhãn:", set(y))
    for it, (x, label) in enumerate(zip(X, y)):
        # Remove false prediction 
        val, ind = model.predict_maxprob(x[None,:]) 
        if ind != label : 
            print(ind, label)
            continue


        x = pytorch_switch(x).detach().numpy()
        params = {
            "x": x,
            "s": args.s,
            "n_queries": args.queries,
            "save_directory": saved_folder,
            "c": x.shape[2],
            "h": x.shape[0],
            "w": x.shape[1],
            "N": args.N,
            "update_loc_period": args.li,
            "mut": args.mut,
            "temp": args.temp
        }
        if mode == "idealW": 
            attack = Attack_idealW(params,it)
            loss = UnTargeted_idealW(model, label)
            print("use ideal world")
        elif mode == "realW": 
            attack = Attack_realW(params,it)
            loss = UnTargeted_realW(model, label)
            print("use real world")
        x_adv = attack.optimise(loss)