import argparse


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-dataset', type=str, default='MKG-W')
    arg.add_argument('-batch_size', type=int, default=512)
    arg.add_argument('-margin', type=float, default=16.0)
    arg.add_argument('-dim', type=int, default=500)
    arg.add_argument('-epoch', type=int, default=1000)
    arg.add_argument('-save', type=str, default='MKG-W-checkpoint')
    arg.add_argument('-neg_num', type=int, default=128)
    arg.add_argument('-learning_rate', type=float, default=1e-5)
    arg.add_argument('-adv_temp', type=float, default=4.0)
    arg.add_argument('-seed', type=int, default=42)
    return arg.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
