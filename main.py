# pylint: disable=E1101,E1102
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd, os, argparse

N_CLASSES = 27

class MyDataset(Dataset):
    def __init__(self, type, base_path, frame_step=1):
        super(MyDataset, self).__init__()
        self._set_path(base_path, type)
        self.indices_df = pd.read_csv(self.indices_path, sep=';', header=None)
        labels_df = pd.read_csv(self.labels_path, sep=';', header=None)
        n_classes = len(labels_df)
        assert(n_classes == N_CLASSES)
        assert(frame_step > 0)
        self.frame_step = frame_step
        self.labels_dict = dict(zip(labels_df.loc[:,0].tolist(), range(n_classes)))

        self.transformer = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ])

    def __len__(self):
        return self.indices_df.shape[0]
    
    def __getitem__(self, index):
        assert(isinstance(index, int))
        v_index, label_str = self.indices_df.loc[index, :].values.tolist()
        return (self._get_video(v_index), self.labels_dict[label_str])

    def _set_path(self, base_path, type):
        self.base_path = base_path
        if type == 'train':
            self.indices_path = os.path.join(self.base_path, r"train.csv")
        elif type == 'validation':
            self.indices_path = os.path.join(self.base_path, r"validation.csv")
        else:
            raise Exception("invalid type: {}".format(type))
        self.labels_path = os.path.join(self.base_path, r"labels.csv")
        self.image_dir_path = os.path.join(self.base_path, r"images")

    def _get_image_path(self, index):
        dir = os.path.join(self.image_dir_path, str(index))
        files = os.listdir(dir)
        r = sorted([os.path.join(dir, f) for f in files if os.path.isfile(os.path.join(dir, f))])
        return r[::self.frame_step]
    
    def _get_video(self, index):
        v = [self.transformer(Image.open(path)) for path in self._get_image_path(index)]
        return torch.stack(v)

def collate_fn(batch):
    # sorting by sequence length descending order
    xs = sorted(list(batch), key=lambda x: len(x[0]), reverse=True)
    ys = list(zip(*xs)) # transpose
    label = torch.tensor(ys[1], dtype=torch.int64)
    packed = pack_sequence(ys[0])
    return packed, label

class Validator():
    def __init__(self, model, dataloader, device, dry_run):
        super(Validator, self).__init__()
        self.model, self.dataloader = model, dataloader
        self.device, self.dry_run = device, dry_run

    def __call__(self):
        model, dataloader, device = self.model, self.dataloader, self.device
        test_loss, correct, total, iter = 0, 0, 0, 0

        model.eval()
        for packed, label in dataloader:
            if iter > 0 and self.dry_run:
                break
            label = label.to(device)
            max_batch_size = packed.batch_sizes[0].item()
            output_shape = [max_batch_size, N_CLASSES]
            output = torch.zeros(output_shape).to(device) # output placeholder
            y = model(packed.to(device))
            offs = 0
            for bs in y.batch_sizes:
                output[0:bs] = y.data[offs:offs+bs]
                offs += bs
            test_loss += F.nll_loss(output, label, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            total += max_batch_size
            iter += 1
            if iter % 10 == 0:
                print("validation processed_data:{}".format(max_batch_size * iter))

        accuracy = correct / total
        test_loss /= total
        print("validation average loss:{:.4f}, accuracy:{:.4f} ({}/{})".format(
            test_loss, accuracy * 100, correct, total))

class Trainer():
    def __init__(self, model, optimizer, dataloader, scheduler,
        last_epoch, max_epoch, device, validator, dry_run):
        self.model, self.optimizer, self.dataloader = model, optimizer, dataloader
        self.scheduler, self.cur_epoch, self.max_epoch = scheduler, last_epoch, max_epoch
        self.device, self.validator, self.dry_run = device, validator, dry_run

    def __call__(self):
        model, optimizer, dataloader = self.model, self.optimizer, self.dataloader
        max_epoch, device, validator = self.max_epoch, self.device, self. validator
        scheduler = self.scheduler

        while self.cur_epoch < max_epoch:
            model.train()
            iter = 0
            correct, total, loss_average = 0, 0, 0
            for packed, label in dataloader:
                if iter > 0 and self.dry_run:
                    break
                max_batch_size = packed.batch_sizes[0].item()
                output_shape = [max_batch_size, N_CLASSES]
                output = torch.zeros(output_shape).to(device) # output placeholder
                optimizer.zero_grad()
                label = label.to(device)
                y = model(packed.to(device))
                offs = 0
                for bs in y.batch_sizes:
                    output[0:bs] = y.data[offs:offs+bs]
                    offs += bs
                loss = F.nll_loss(output, label)
                loss_average += loss.item()
                loss.backward()
                optimizer.step()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()
                total += max_batch_size

                iter += 1
                if iter % 10 == 0:
                    accuracy = 100 * correct / total
                    args = [self.cur_epoch,
                        max_batch_size * iter,
                        loss_average / 10, accuracy,
                        correct, total]
                    print("training epoch:{} processed_data:{} loss:{:.4f} accuracy:{:.4f} ({}/{})".format(*args))
                    correct, total, loss_average = 0, 0, 0

            scheduler.step()
            self.cur_epoch += 1
            torch.save({
                'last_epoch' : self.cur_epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict()
            }, "checkpoint.pt")
            validator()

class PlasticNet(nn.Module):
    def __init__(self, isize, hsize):
        super(PlasticNet, self).__init__()
        self.hsize, self.isize  = hsize, isize
        self.i2h = torch.nn.Linear(isize, hsize)
        self.w =  torch.nn.Parameter(.001 * torch.rand(hsize, hsize))
        self.alpha =  torch.nn.Parameter(.001 * torch.rand(hsize, hsize))
        self.h2mod = torch.nn.Linear(hsize, 1)
        self.modfanout = torch.nn.Linear(1, hsize)

    def forward(self, x, state=None):
        assert(isinstance(x, PackedSequence))
        max_batch_size = x.batch_sizes[0].item()
        if state is None:
            state = self.initial_state(max_batch_size, x.data.device)
        offs = 0
        r = []
        for bs in x.batch_sizes:
            h_pred = state[0][0:bs]
            hebb = state[1][0:bs]
            inputs = x.data[offs:offs+bs]

            h = torch.tanh(self.i2h(inputs) + h_pred.unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze(1))
            deltahebb = torch.bmm(h_pred.unsqueeze(2), h.unsqueeze(1))
            myeta = torch.tanh(self.h2mod(h)).unsqueeze(2)
            myeta = self.modfanout(myeta)
            clipval = 2.0
            hebb = torch.clamp(hebb + myeta * deltahebb, min=-clipval, max=clipval)

            state = (h, hebb)
            r.append(h)
            offs += bs
        return PackedSequence(torch.cat(r, dim=0), x.batch_sizes), state
        
    def initial_state(self, batch_size, device):
        h_size = self.hsize
        h = Variable(torch.zeros(batch_size, h_size), requires_grad=False).to(device)
        hebb = Variable(torch.zeros(batch_size, h_size, h_size), requires_grad=False).to(device)
        return h, hebb

class PlasticLSTM(nn.Module):
    def __init__(self, isize, hsize):
        super(PlasticLSTM, self).__init__()
        self.isize, self.hsize = isize, hsize
        self.x2fioj = nn.Linear(isize, 4 * hsize)
        self.h2fioj = nn.Linear(hsize, 4 * hsize)
        self.h2mod = nn.Linear(hsize, 1)
        self.modfanout = nn.Linear(1, hsize)
        self.alpha = nn.Parameter(0.0001 * torch.rand((hsize, hsize)))

    def forward(self, x, state=None):
        assert(isinstance(x, PackedSequence))
        hsize = self.hsize
        max_batch_size = x.batch_sizes[0].item()
        if state is None:
            state = self.initial_state(max_batch_size, x.data.device)
        offs = 0
        r = []
        for bs in x.batch_sizes:
            h_pred, c_pred, hebb = state[0][0:bs], state[1][0:bs], state[2][0:bs]
            fioj = self.x2fioj(x.data[offs:offs+bs]) + self.h2fioj(h_pred)
            f = torch.sigmoid(fioj[:,:hsize])
            i = torch.sigmoid(fioj[:,hsize:2*hsize])
            o = torch.sigmoid(fioj[:,2*hsize:3*hsize])
            j = torch.tanh(fioj[:,3*hsize:] + h_pred.unsqueeze(1).bmm(torch.mul(self.alpha, hebb)).squeeze(1))
            c = torch.mul(f, c_pred) + torch.mul(i, j)
            h = torch.mul(o, torch.tanh(c))

            delta_hebb = torch.bmm(h_pred.unsqueeze(2), j.unsqueeze(1))
            myeta = torch.tanh(self.h2mod(c)).unsqueeze(2)
            myeta = self.modfanout(myeta).squeeze().unsqueeze(1)
            clipval = 2.0
            hebb = torch.clamp(hebb + myeta * delta_hebb, min=-clipval, max=clipval)
            state = (h, c, hebb)
            r.append(h)
            offs += bs
        return PackedSequence(torch.cat(r, dim=0), x.batch_sizes), state

    def initial_state(self, batch_size, device):
        hsize = self.hsize
        h = Variable(torch.zeros(batch_size, hsize), requires_grad=False).to(device)
        c = Variable(torch.zeros(batch_size, hsize), requires_grad=False).to(device)
        hebb = Variable(torch.zeros(batch_size, hsize, hsize), requires_grad=False).to(device)
        return h, c, hebb

class Net(nn.Module):
    def __init__(self, mode='backpropamine'):
        super(Net, self).__init__()
        self.mode = mode
        self.mobilenet = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        self.mobilenet.classifier = nn.Identity()
        self.mobilenet = nn.DataParallel(self.mobilenet)

        if mode == 'backpropamine':
            self.rnn = PlasticLSTM(1280, 1000)
        elif mode == 'LSTM':
            self.rnn = nn.LSTM(1280, 1000)
        else:
            raise Exception("invalid mode: {}".format(self.mode))

        self.fc = nn.Linear(1000, N_CLASSES)

    def forward(self, x):
        assert(isinstance(x, PackedSequence))
        x = PackedSequence(self.mobilenet(x.data), x.batch_sizes)
        x, _ = self.rnn(x)
        return PackedSequence(F.log_softmax(self.fc(x.data), dim=1), x.batch_sizes)

def main():
    parser = argparse.ArgumentParser(description='20bn-jester-v1 Gesture Classification with Backpropamine')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    #parser.add_argument('--validation-batch-size', type=int, default=1000, metavar='N',
    #                    help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num-workers', type=int, default=0, metavar='W',
                        help='number of workers for data loading (default: 0)')
    parser.add_argument('--lr', type=float, default=1., metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dataset-dir', type=str, default=r"./dataset", metavar='D',
                        help='dataset place (default: ./dataset)')
    #parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                    help='how many batches to wait before logging training status')
    #parser.add_argument('--save-model', action='store_true', default=False,
    #                    help='For Saving the current Model')
    parser.add_argument('--no-resume', action='store_true', default=False,
                        help='switch to disables resume')
    parser.add_argument('--use-lstm', action='store_true', default=False,
                        help='switch to use LSTM module instead of backpropamine')
    parser.add_argument('--frame-step', type=int, default=2, metavar='FS',
                        help='step of video frames extraction (default: 2)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    torch.manual_seed(args.seed)

    train_data = MyDataset('train', args.dataset_dir, frame_step=args.frame_step)
    validation_data = MyDataset('validation', args.dataset_dir, frame_step=args.frame_step)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True,
        shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    validation_dataloader = DataLoader(validation_data, batch_size=args.batch_size, drop_last=True,
        shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

    resume = not args.no_resume

    if resume:
        try:
            checkpoint = torch.load("checkpoint.pt")
        except FileNotFoundError:
            resume = False

    mode = 'LSTM' if args.use_lstm else 'backpropamine'
    model = Net(mode=mode).to(device)
    optimizer = Adadelta(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    last_epoch, max_epoch = 0, args.epochs

    if resume:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        last_epoch = checkpoint['last_epoch']
    
    validator = Validator(model, validation_dataloader, device, args.dry_run)
    trainer = Trainer(model, optimizer, train_dataloader, scheduler,
        last_epoch, max_epoch, device, validator, args.dry_run)

    print(vars(args))
    trainer()
    print("finish.")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
