import argparse
import auxil
from hyper_pytorch import *
import torch
import torch.nn.parallel
from torchvision.transforms import *
import models.DCAENet as DCAE
import random
import os
import time
import scipy.io as sio

seed=100
random.seed(seed)  # Python random module.
np.random.seed(seed)  # Numpy module.
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed) # 
torch.cuda.manual_seed(seed) # 
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.use_deterministic_algorithms(True)


def load_hyper(args):
    data, mask = auxil.loadData(args.dataset, num_components=args.components, standard = args.standard)
    row, col, band = data.shape
    x_train = data[np.newaxis, :]
    data2d = np.reshape(x_train, (row*col, band))
    train_hyper = HyperData(np.transpose(x_train, (0, 3, 1, 2)).astype("float32"))
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.tr_bsize, shuffle=False)
    return train_loader, data2d, band, mask


def train(trainloader, args, model, criterion1, criterion2, weight_init2, loss_last, epoch, optimizer, scheduler, use_cuda):
    model.train()
    losses = np.zeros((len(trainloader))) 
    accs   = np.zeros((len(trainloader))) 
    for batch_idx, (input) in enumerate(trainloader):
        f_w1 = torch.ones([input.shape[2], input.shape[3]])
        f_w1 = f_w1/torch.mean(f_w1)
        if use_cuda:
            input, weight_init2= input.cuda(), weight_init2.cuda()
            input, weight_init2= input.float(), weight_init2.float()
            f_w1 = f_w1.cuda()

        outputs = model(input, args)
        
        # add an adpative weight to compute loss
        ##############################################
        if loss_last <0.1:
            if (epoch+1) % 50== 0:
               res = input-outputs
               f_w1 = torch.sqrt(torch.sum(torch.square(res),axis=1))
               f_w1 = torch.max(f_w1)-f_w1
               f_w1 = (f_w1-torch.min(f_w1))/(torch.max(f_w1)-torch.min(f_w1))
               f_w1 = f_w1.squeeze(0)
               f_w1 = f_w1/torch.mean(f_w1)
        #############################################

        model_dict = model.state_dict()
        weight = model_dict['decoderlayer1.0.weight']

        loss = criterion1(f_w1*input, f_w1*outputs) 
   
        if args.lamda!=0:
           loss = loss + args.lamda*criterion2(weight_init2, weight)    

        losses[batch_idx] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return (np.average(losses))


def test(test_loader, args, mask, model, criterion, use_cuda, draw = False):
    model.eval()
    accs   = np.zeros((len(test_loader))) 
    losses = np.zeros((len(test_loader))) 
    for batch_idx, (input) in enumerate(test_loader):
        if use_cuda:
            input = input.cuda()
            input = input.float()

        outputs= model(input, args)

        loss= criterion(outputs, input)
        losses[batch_idx] = loss.item()
        anomaly_map, PD, PF, accs[batch_idx] = auxil.accuracy(outputs.data, input.data, mask, draw)

        test_result = {
                        'anomaly_map': anomaly_map,
                        'PD': PD,
                        'PF': PF,
                        'AUC' : np.average(accs)
                    }

        return (np.average(losses), test_result)


def main():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--tr_bsize', default=1, type=int, help='mini-batch train size (default: 100)')
    parser.add_argument('--KernelS', default=3, type=int, help='kernel size')
    parser.add_argument('--components', default=False, type=int, help='dimensionality reduction')
    parser.add_argument('--standard', default=False, type=int, help='Is the data standardized')
    parser.add_argument('--thre', default=1e-6, type=float, help='the threshold of stopping training')
    parser.add_argument('--dataset', default='ABU_A4', type=str, help='dataset (options: HYDICE, PC, SD)')
    parser.add_argument('--en_dim',  default=4, type=int, help='The number of the clusters')
    parser.add_argument('--norm',  default='l1', type=str, help='type of reconstruction loss')
    parser.add_argument('--guide',  default=True, type=int, help='if using background guidance')
    parser.add_argument('--initial', default=True, type=int, help='Is the training guided by background initialization')
    parser.add_argument('--lamda', default=0, type=float, help='the tradeoff of loss of background guidance')
    parser.add_argument('--NLC', default=True, type=int, help='Non-local Convolution')
    parser.add_argument('--DC', default=True, type=int, help='Deformable Convolution')

    args = parser.parse_args()
    train_loader, data2d, band, mask = load_hyper(args)

    # Is the network training guided by background？
    ###########################
    if args.en_dim==0:
        args.en_dim = auxil.estimate_Nclust(data2d)

    if args.guide:
        weight_guide, args = auxil.KClus(data2d, args)
    else:
        weight_guide = torch.zeros([band, args.en_dim, 1, 1])
    ##########################
        
    # Use CUDA and define model
    use_cuda = torch.cuda.is_available()
    if use_cuda: torch.backends.cudnn.benchmark = False
    model = DCAE.DCAENet(band, args.en_dim, args.KernelS)
    model.apply(auxil.weights_init)
    model = model.float()
    
    #Is the weight of decoder initialized by background？
    ################################# 
    if args.initial:
        model_dict = model.state_dict()
        model_dict['decoderlayer1.0.weight'] = weight_guide
        model.load_state_dict(model_dict)
    #################################
    
    if use_cuda: model = model.cuda()
    optimizer = torch.optim.Adam \
         (model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.7) 
    # summary(model, (175, 80, 100))
    
    # define loss function
    if args.norm == 'l1':
       criterion1 = torch.nn.L1Loss()
    elif args.norm == 'l2':
       criterion1 = torch.nn.MSELoss()
    elif args.norm == 'Sl1':
       criterion1 = torch.nn.SmoothL1Loss()
    criterion2 = torch.nn.MSELoss()
   
    # variables initialization 
    loss_np = np.zeros((1, 30), dtype=np.float32)
    loss_last = 100
    mean_loss =1
    best_acc = 0
    loss_value = np.zeros(args.epochs)
    acc_value = np.zeros(args.epochs)


    for epoch in range(args.epochs):

        train_loss= train(train_loader, args, model, criterion1, criterion2, weight_guide, loss_last, epoch, optimizer, scheduler, use_cuda)
        test_loss, test_result = test(train_loader, args, mask, model, criterion1, use_cuda)
        test_acc = test_result['AUC']
        
        if epoch % 10== 0:
           print("EPOCH", epoch, "TRAIN LOSS", train_loss, end=',')
           print("TEST LOSS", test_loss, "ACCURACY", test_acc)

        # 
        loss_value[epoch] = train_loss
        acc_value[epoch] = test_acc

        # save model whose result is the best in the epochs
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'loss': train_loss,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, "best_model.pth.tar")

            if epoch >= 1:
               index = epoch-int(epoch/30)*30
               loss_np[0][index-1] = abs(train_loss-loss_last)
               if epoch % 30 == 0:
                  mean_loss = np.mean(loss_np)
                

        loss_last = train_loss

        if epoch == args.epochs-1 or mean_loss < args.thre:
            checkpoint = torch.load("best_model.pth.tar")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            _, test_result = test(train_loader, args, mask, model, criterion1, use_cuda, draw = True)

            test_acc = test_result['AUC']
            print("FINAL: LOSS", checkpoint['loss'], "ACCURACY", checkpoint['best_acc'])

            return test_result
        

if __name__ == '__main__':

    start = time.perf_counter()

    result = main()

    end = time.perf_counter()
    time_DCAE = end-start

    sio.savemat('A4_DCAE.mat', {'A_AutoAD': result['anomaly_map'], 'PD_AutoAD': result['PD'], 'PF_AutoAD': result['PF'],
                                       'AUC_AutoAD': result['AUC'], 'time_AutoAD': time_DCAE})

    print("AUC: ", result['AUC'], "Time: ", time_DCAE)