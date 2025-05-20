from torchvision import transforms
from src.model import SignNN 
from src.loss_function import UnTargeted_idealW, UnTargeted_realW
from src.attack import Attack_idealW, Attack_realW
import argparse
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

    parser.add_argument("--image_dir", type=str, default="data/TEST/0/000_1_0004_1_j.png", help="Image File path")
    parser.add_argument("--true_label", type=int,default=0, help="Number of the correct label of ImageNet inputted image")
    parser.add_argument("--save_directory", type=str,default=".", help="Where to store the .npy files with the results")
    args = parser.parse_args()
    mode = args.mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignNN().to(device)
    model.eval() 
    model.load_state_dict(torch.load('saved_model/best_f1.pt'))

    image_dir = args.image_dir
    x_test = transform(Image.open(image_dir))
    x_test_4d = x_test[None,:]
    ori_conf,ori_pred = model.predict_maxprob(x_test_4d)
    print(f"Original prediction: {ori_pred}, with {ori_conf} confidence")
    assert ori_pred == args.true_label, "Not need to attack"

    x = pytorch_switch(x_test).detach().numpy()
    params = {
        "x": x,
        "s": args.s,
        "n_queries": args.queries,
        "save_directory": args.save_directory ,
        "c": x.shape[2],
        "h": x.shape[0],
        "w": x.shape[1],
        "N": args.N,
        "update_loc_period": args.li,
        "mut": args.mut,
        "temp": args.temp, 
    }
    if mode == "idealW": 
        attack = Attack_idealW(params)
        loss = UnTargeted_idealW(model, args.true_label)
        print("use ideal world")
    elif mode == "realW": 
        attack = Attack_realW(params)
        loss = UnTargeted_realW(model, args.true_label)
        print("use real world")
    x_adv = attack.optimise(loss)
    x_adv = to_pytorch(x_adv)[None,:]
    lat_conf, lat_pred = model.predict_maxprob(x_adv) 
    print(f"Latter prediction: {lat_pred}, with {lat_conf} confidence")