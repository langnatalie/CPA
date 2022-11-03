import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='exp',
                        help="the name of the current experiment")
    parser.add_argument('--eval', action='store_true',
                        help="weather to perform inference of training")

    # data arguments
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist', 'cifar10'],
                        help="dataset to use (mnist or cifar)")
    parser.add_argument('--norm_mean', type=float, default=0.5,
                        help="normalize the data to norm_mean")
    parser.add_argument('--norm_std', type=float, default=0.5,
                        help="normalize the data to norm_std")
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help="trainset batch size")
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help="testset batch size")

    # federated arguments
    parser.add_argument('--model', type=str, default='linear',
                        choices=['cnn2', 'cnn3', 'mlp', 'linear'],
                        help="model to use (cnn, mlp)")
    parser.add_argument('--num_users', type=int, default=1000,
                        help="number of users participating in the federated learning")
    parser.add_argument('--local_epochs', type=int, default=1,
                        help="number of local epochs")
    parser.add_argument('--local_iterations', type=int, default=1,
                        help="number of local iterations instead of local epoch")
    parser.add_argument('--global_epochs', type=int, default=30,
                        help="number of global epochs")
    parser.add_argument('--aggregation_method', default='FedAvg',
                        choices=['CPA', 'FedAvg'],
                        help="centralized or federated learning")
    parser.add_argument('--threshold', type=float, default=None,
                        help="zero the weight if users values summation is beneath the threshold")

    # privacy arguments
    parser.add_argument('--privacy', default=None,
                        choices=['RR', 'Laplace', None],
                        help="select privacy mechanism type")
    parser.add_argument('--epsilon', type=float, default=1,
                        help="privacy budget (epsilon)")

    # quantization arguments
    parser.add_argument('--compression', default=None,
                        choices=['scalarQ', 'signSGD', None],
                        help="select compression type")
    parser.add_argument('--R', type=int, default=3,
                        help="compression rate (number of bits per sample)")
    parser.add_argument('--gamma', type=float, default=0.05,
                        help="quantizer dynamic range")

    # robustness arguments
    parser.add_argument('--malicious', default=None,
                        choices=['ones', 'flip', None],
                        help="select malicious users manipulation type")
    parser.add_argument('--malicious_users_percent', type=float, default=0.1,
                        help="select the percentage of malicious users")


    # learning arguments
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help="optimizer to use (sgd or adam)")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate is 0.01 for cnn and 0.1  for linear")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="momentum")
    parser.add_argument('--lr_scheduler', action='store_false',
                        help="reduce the learning rat when val_acc has stopped improving (increasing)")
    parser.add_argument('--device', type=str, default='cuda:0',
                        choices=['cuda:0', 'cuda:1', 'cpu'],
                        help="device to use (gpu or cpu)")
    parser.add_argument('--seed', type=float, default=1234,
                        help="manual seed for reproducibility")

    args = parser.parse_args()
    return args
