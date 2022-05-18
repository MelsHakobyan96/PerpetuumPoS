import argparse

from train import train_model


parser = argparse.ArgumentParser(
    description='Our approach to imitation learning.')

# -- Training related arguments --
parser.add_argument('--epochs', type=int, required=False, default=100,
                    help='The number of epochs to train the model.')
parser.add_argument('--cuda', type=bool, required=False, default=False,
                    help='Use GPU or not')
parser.add_argument('--checkpoint', type=str, required=False,
                    help='Load from checkpoint.')

# -- Data related arguments --
parser.add_argument('--data_dir_path', type=str,
                    help='The directory of the data.')
parser.add_argument('--train_csv_path', type=str,
                    help='The path to the csv file for training.')
parser.add_argument('--valid_csv_path', type=str,
                    help='The path to the csv files for validation.')
parser.add_argument('--test_csv_path', type=str, required=False,
                    help='The path to the csv files for testing.')
parser.add_argument('--batch_size', type=int, required=False, default=1,
                    help='The batch size of the training data.')

# -- Model related arguments --
parser.add_argument('--out_channels', type=int, required=False, default=8,
                    help='The output channel number of conv layers.')
parser.add_argument('--kernel_size', type=int, required=False, default=3,
                    help='The kernel size of each conv layer.')
parser.add_argument('--gru_hidden_size', type=int, required=False, default=128,
                    help='The hidden size for GRU.')
parser.add_argument('--gru_num_layers', type=int, required=False, default=2,
                    help='The number of GRU layers in the model.')

args = parser.parse_args()

if __name__ == '__main__':
    train_model(args)
