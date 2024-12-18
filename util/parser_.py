import argparse
from util.colors import yellow, magenta

def get_argparse():
    """
    parse command line arguments
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='mvtec_ad', choices=['MVTec_AD', 'MVTec_LOCO_AD'])
    parser.add_argument('--subdataset', dest='subdataset', type=str, help='One of 15 sub-datasets of Mvtec AD or 5 sub-datasets of Mvtec LOCO')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='output/1')
    parser.add_argument('--model_size', dest='model_size', type=str, default='small', choices=['small', 'medium'])
    parser.add_argument('--weights', dest='weights', type=str, default='models/teacher_small.pth')
    parser.add_argument('--imagenet_train_path', dest='imagenet_train_path', type=str, default='none', help='Set to "none" to disable ImageNet pretraining penalty. Or see README.md to download ImageNet and set to ImageNet path')
    parser.add_argument('--train_steps', dest='train_steps', type=int, default=70000)
    
    args = parser.parse_args()
    
    print(f"dataset : {yellow(args.dataset)}")
    print(f"subdataset : {magenta(args.subdataset)}")
    print(f"output_dir : {yellow(args.output_dir)}")
    print(f"model_size : {magenta(args.model_size)}")
    print(f"weights : {yellow(args.weights)}")
    print(f"imagenet_train_path : {magenta(args.imagenet_train_path)}")
    print(f"train_steps : {yellow(args.train_steps)}")
    
    return parser.parse_args()