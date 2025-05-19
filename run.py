from torchvision import transforms
from src.model import SignNN 
from src.loss_function import UnTargeted_idealW, UnTargeted_realW
from src.attack import Attack_idealW, Attack_realW
import argparse
from src.utils import *
from PIL import Image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--temp", type=float, default=300.)
    parser.add_argument("--mut", type=float, default=0.3)
    parser.add_argument("--s", type=int, default=20)
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--li", type=int, default=4)

    parser.add_argument("--image_dir", type=str, default="data/TEST/0/000_1_0003_1_j.png", help="Image File path")
    parser.add_argument("--true_label", type=int,default=0, help="Number of the correct label of ImageNet inputted image")
    parser.add_argument("--save_directory", type=str,default="result/ex1", help="Where to store the .npy files with the results")
    args = parser.parse_args()

    model = SignNN()
    model.eval() 
    model.load_state_dict(torch.load('saved_model/best_f1.pt'))
    image_dir = args.image_dir
    x_test = transform(Image.open(image_dir))

    loss = UnTargeted_idealW(model, args.true_label)
    x = pytorch_switch(x_test).detach().numpy()
    params = {
        "x": x,
        "s": args.s,
        "n_queries": args.queries,
        "save_directory": args.save_directory + ".npy",
        "c": x.shape[2],
        "h": x.shape[0],
        "w": x.shape[1],
        "N": args.N,
        "update_loc_period": args.li,
        "mut": args.mut,
        "temp": args.temp
    }
    attack = Attack_idealW(params)
    attack.optimise(loss)