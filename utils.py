import os
from statistics import mean
import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
from scipy.interpolate import interp1d

def data(args):
    if args.data == 'mnist':
        train_data = datasets.MNIST('./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                    ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
    else:
        train_data = datasets.CIFAR10('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                      ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
    return train_data, test_loader


def data_split(data, amount, args):
    # split train, validation
    train_data, val_data = torch.utils.data.random_split(data, [len(data) - amount, amount])
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False)

    # input, output sizes
    in_channels, dim1, dim2 = data[0][0].shape  # images are dim1 x dim2 pixels
    input = dim1 * dim2 if args.model == 'mlp' or args.model == 'linear' else in_channels
    output = len(data.classes)  # number of classes

    return input, output, train_data, val_loader


def train_one_epoch(train_loader, model, optimizer,
                    creterion, device, iterations):
    model.train()
    losses = []
    if iterations is not None:
        local_iteration = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # send to device
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = creterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if iterations is not None:
            local_iteration += 1
            if local_iteration == iterations:
                break
    return mean(losses)


def test(test_loader, model, creterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)  # send to device

        output = model(data)
        test_loss += creterion(output, label).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


def initializations(args):
    #  reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    #  documentation
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')

    best_val_acc = np.NINF
    path_best_model = 'checkpoints/' + args.exp_name + '/model.best.t7'

    return boardio, textio, best_val_acc, path_best_model


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def plot_convergence_curves():
    iterations = 30
    x = np.arange(iterations)
    X_ = np.linspace(x.min(), x.max(), 150)
    titles = []

    initial_path = f'./checkpoints/convergence/mnist/'
    final_path = f'/val_acc_list.npy'

    titles.append('vanilla FL')
    y = np.load(initial_path + f'FL' + final_path)[:iterations]
    cubic_interpolation_model = interp1d(x, y, kind="cubic")
    Y_ = cubic_interpolation_model(X_)
    plt.plot(X_, Y_, '-')

    titles.append(f'Laplace')
    y = np.load(initial_path + f'Laplace' + final_path)[:iterations]
    cubic_interpolation_model = interp1d(x, y, kind="cubic")
    Y_ = cubic_interpolation_model(X_)
    plt.plot(X_, Y_, ':')

    titles.append(f'signRR')
    y = np.load(initial_path + f'signRR' + final_path)[:iterations]
    cubic_interpolation_model = interp1d(x, y, kind="cubic")
    Y_ = cubic_interpolation_model(X_)
    plt.plot(X_, Y_, linestyle=(0, (3, 1, 1, 1, 1, 1)))

    titles.append(f'JoPEQ')
    y = np.load(initial_path + f'JoPEQ' + final_path)[:iterations]
    cubic_interpolation_model = interp1d(x, y, kind="cubic")
    Y_ = cubic_interpolation_model(X_)
    plt.plot(X_, Y_, linestyle=(0, (3, 1, 1, 1, 1, 1)))

    titles.append(f'CPA')
    y = np.load(initial_path + f'CPA/' + final_path)[:iterations]
    cubic_interpolation_model = interp1d(x, y, kind="cubic")
    Y_ = cubic_interpolation_model(X_)
    plt.plot(X_, Y_, '--')

    titles.append(f'CPAwoRR')
    y = np.load(initial_path + f'CPAwoRR/eps0p5/' + final_path)[:iterations]
    cubic_interpolation_model = interp1d(x, y, kind="cubic")
    Y_ = cubic_interpolation_model(X_)
    plt.plot(X_, Y_, '-.')


    plt.legend(titles)
    plt.xlabel('Global iteration')
    plt.ylabel('Validation set accuracy')
    #plt.title('FL with ' + f'{users}' + ' edge users')

    plt.grid()
    plt.savefig(initial_path + f'convergence.pdf', transperent=True, bbox_inches='tight')


def plot_SNR_curves():
    x = np.arange(start=0.5, stop=3, step=0.5)
    X_ = np.linspace(x.min(), x.max(), 45)
    fig, ax = plt.subplots()
    # y1 = [-6.494, -8.872, -9.472, -9.884, -10.025, -10.432, -10.07, -10.084, -10.48]
    # y = [-10.021, -4.869, 4.372, 3.151]
    # y = [-10.32, -4.497, 6.239, 10.908, 13.368, 18.794]
    y = [3.128, 11.338, 14.132, 17.875, 18.838]
    cubic_interpolation_model = interp1d(x, y, kind="cubic")
    Y_ = cubic_interpolation_model(X_)
    cpa1000, = ax.plot(X_[:], Y_[:], linestyle=(0, (3, 1, 1, 1)), label=r'CPA, $K=1000$')

    #y = [-20.606, -16.448, -6.731, -2.312, 2.055, 2.706]
    y = [-5.236, 2.274, 6.49, 8.726, 9.409]
    cubic_interpolation_model = interp1d(x, y, kind="cubic")
    Y_ = cubic_interpolation_model(X_)
    cpa100, = ax.plot(X_[:], Y_[:], linestyle=(0, (3, 1, 1, 1, 1, 1)), label=r'CPA, $K=100$')

    first_legend = ax.legend(handles=[cpa1000, cpa100], loc='upper left', bbox_to_anchor=(0.67, 0.15, 0.5, 0.5))

    # y2 = [-92.848, -83.028, -77.585, -73.893, -70.997, -68.641, -66.245, -64.113, -62.258]
    # y = [-60.674, -47.134, -40.226, -33.966]
    # y = [-53.15, -39.646, -33.287, -27.402, -23.047, -20.751]
    y = [-4.442, 6.712, 16.119, 20.928, 23.284]
    cubic_interpolation_model = interp1d(x, y, kind="cubic")
    Y_ = cubic_interpolation_model(X_)
    laplace1000, = ax.plot(X_[:], Y_[:], '-.', label=r'Laplace, $K=1000$')

    # y = [-70.997, -60.674, -52.892, -47.134, -43.669, -40.226]
    y = [-24.77, -13.641, -5.227, -0.547, 3.109]
    cubic_interpolation_model = interp1d(x, y, kind="cubic")
    Y_ = cubic_interpolation_model(X_)
    laplace100, = ax.plot(X_[:], Y_[:], ':', label=r'Laplace, $K=100$')

    second_legend = ax.legend(handles=[laplace1000, laplace100], loc='center right', bbox_to_anchor=(0.5, 0.15, 0.5, 0.5))

    # y3 = [-39.852, -40.094, -38.092, -36.602, -40.46, -38.943, -40.912, -42.295, -41.397]
    # y = [-42.554, -49.86, -52.418, -54.169]
    # y = [-31.337, -41.452, -48.001, -50.686, -53.547, -54.524]
    y = [-30.176, -39.552, -45.359, -47.131, -50.543]
    cubic_interpolation_model = interp1d(x, y, kind="cubic")
    Y_ = cubic_interpolation_model(X_)
    signSGDRR1000, = ax.plot(X_[:], Y_[:], '--', label=r'signRR, $K=1000$')

    # y = [-40.443, -42.574, -49.397, -50.356, -52.31, -53.549]
    y = [-39.036, -40.637, -46.903, -47.455, -48.909]
    cubic_interpolation_model = interp1d(x, y, kind="cubic")
    Y_ = cubic_interpolation_model(X_)
    signSGDRR100, = ax.plot(X_[:], Y_[:], '-', label=r'signRR, $K=100$')

    third_legend = ax.legend(handles=[signSGDRR1000, signSGDRR100], loc='lower left', bbox_to_anchor=(0.63, 0.14, 0.5, 0.5))
    plt.gca().add_artist(second_legend)
    plt.gca().add_artist(first_legend)

    plt.xlabel(r'$\epsilon$')
    plt.ylabel('SNR [dB]')
    plt.autoscale(enable=True, axis='y')
    # plt.title('FL with ' + f'{users}' + ' edge users')

    plt.grid()
    plt.savefig(f'./checkpoints/SNR/SNR_1000and100_users.pdf', transperent=True, bbox_inches='tight')


if __name__ == '__main__':
    #plot_convergence_curves()
    # plot_SNR_curves()