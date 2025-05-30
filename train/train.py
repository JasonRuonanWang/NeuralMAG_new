import os
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import stats
import logging

import torch
import torch.optim as optimizer

from Unet import UNet
from data_load import dataset_prepare
from utils import AverageMeter, train_loss_compute, eval_loss_compute, dataug

def train(epoch, model, optim, schduler, train_dataloader1, train_dataloader2, train_dataloader3, ex_path):
    model.train()
    Loss = AverageMeter()
    Loss11 = AverageMeter()
    Loss12 = AverageMeter()
    Loss21 = AverageMeter()
    Loss22 = AverageMeter()
    Loss31 = AverageMeter()
    Loss32 = AverageMeter()
    if args.dataug==True:
        pbar = tqdm(total=round(1.1*len(train_dataloader1.dataset)))
    else:
        pbar = tqdm(total=len(train_dataloader1.dataset))
    
    # start to train
    for batch_idx, (batch1, batch2, batch3)  in enumerate(zip(train_dataloader1, train_dataloader2, train_dataloader3)):

        x1, y1 = batch1
        x2, y2 = batch2
        x3, y3 = batch3
        x1, y1, x2, y2, x3, y3 = x1.to(device), y1.to(device), x2.to(device), y2.to(device), x3.to(device), y3.to(device)  

        if args.dataug==True:
            x1, y1 = dataug(x1,y1)
            x2, y2 = dataug(x2,y2)
            x3, y3 = dataug(x3,y3)
        size1 = x1.size(0)
        size2 = x2.size(0)
        size3 = x3.size(0)
        is_last_batch = (batch_idx == len(train_dataloader1) - 1)
        loss1, loss11 = train_loss_compute(x1, y1, model, Loss11, Loss12, epoch, is_last_batch, 32, ex_path, "train")
        loss2, loss21 = train_loss_compute(x2, y2, model, Loss21, Loss22, epoch, is_last_batch, 64, ex_path, "train")    
        loss3, loss31 = train_loss_compute(x3, y3, model, Loss31, Loss32, epoch, is_last_batch, 128, ex_path, "train")
        loss = loss1 + loss2 + loss3

        loss.backward()
        optim.step()
        optim.zero_grad()
        Loss.update( ((loss11.mean() + loss21.mean() + loss31.mean())/3).item(), size1+size2+size3)

        pbar.update(size1)
        pbar.set_description('epoch {} Loss {:.1f} Loss1 {:.1f} / {:.3f} Loss2 {:.1f} / {:.3f} Loss3 {:.1f} / {:.3f}'.format(
                               epoch, Loss.avg, Loss11.avg, Loss12.avg, Loss21.avg, Loss22.avg, Loss31.avg, Loss32.avg
                                
                               )
                            )
    pbar.close()
    if epoch > 150:
        scheduler.step()

    return Loss.avg

def eval(epoch, model, dataloader1, dataloader2, dataloader3, dataloader4, ex_path):

    model.eval()
    Loss = AverageMeter()
    Loss1 = AverageMeter()
    Loss2 = AverageMeter()
    Loss3 = AverageMeter()
    Loss4 = AverageMeter()

    pbar = tqdm(total=len(dataloader1.dataset))
    # start to train
    for batch_idx, (batch1, batch2, batch3, batch4)  in enumerate(zip(dataloader1, dataloader2, dataloader3, dataloader4)):
        x1, y1 = batch1
        x2, y2 = batch2
        x3, y3 = batch3
        x4, y4 = batch4
        x1, y1, x2, y2, x3, y3, x4, y4 = x1.to(device), y1.to(device), x2.to(device), y2.to(device), x3.to(device), y3.to(device), x4.to(device), y4.to(device)

        size1 = x1.size(0)
        size2 = x2.size(0)
        size3 = x3.size(0)
        size4 = x4.size(0)
        is_last_batch = (batch_idx == len(dataloader1) - 1)
        with torch.no_grad():
            loss1= eval_loss_compute(x1, y1, model, Loss1, epoch, is_last_batch, 32, ex_path, "eval")
            loss2= eval_loss_compute(x2, y2, model, Loss2, epoch, is_last_batch, 64, ex_path, "eval")
            loss3= eval_loss_compute(x3, y3, model, Loss3, epoch, is_last_batch, 128, ex_path, "eval")
            loss4= eval_loss_compute(x4, y4, model, Loss4, epoch, is_last_batch, 256, ex_path, "eval")
        Loss1.update( (loss1).item(), size1)
        Loss2.update( (loss2).item(), size2)
        Loss3.update( (loss3).item(), size3)
        Loss4.update( (loss4).item(), size4)
        Loss.update( ((loss1.mean()+loss2.mean()+loss3.mean()+loss4.mean())/4).item(), size1+size2+size3+size4) 
        pbar.update(size1)
        pbar.set_description('Eval: epoch {} Loss {:.1f} Loss1 {:.1f} Loss2 {:.1f}  Loss3 {:.1f} Loss4 {:.1f}'.format(
                              epoch, Loss.avg, Loss1.avg, Loss2.avg, Loss3.avg, Loss4.avg
                               )
        )

    pbar.close()
    return Loss1.avg, Loss2.avg, Loss3.avg, Loss4.avg, Loss.avg

def save_best_model(model, epoch, loss_test, best_loss, ex_path):
    model_path = os.path.join(ex_path, "ckpt/")
    os.makedirs(model_path, exist_ok=True)
    
    if loss_test < best_loss:
        print('loss_test: {:.1f} < best_loss: {:.1f} \n'.format(loss_test, best_loss))
        best_loss = loss_test
        model_to_access = model.module if isinstance(model, torch.nn.DataParallel) else model
        best_model_state_dict = model_to_access.state_dict()
        torch.save(best_model_state_dict, f"{model_path}/best_model_ep{epoch}_{best_loss:.1f}.pt")
    return best_loss

def draw_loss(epoch_list, loss_train_list, loss_test_list1, loss_test_list2, loss_test_list3, loss_test_list4, ex_path, ex):
    plt.clf()
    plt.plot(epoch_list, loss_train_list, 'r-', alpha=1, label='train_32_64_128')
    plt.plot(epoch_list, loss_test_list1, 'c-',  alpha=1, label='test_32')
    plt.plot(epoch_list, loss_test_list2, 'g-',  alpha=1, label='test_64')
    plt.plot(epoch_list, loss_test_list3, 'b-',  alpha=1, label='test_128')
    plt.plot(epoch_list, loss_test_list4, 'm-',  alpha=1, label='test_256')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss-log')
    plt.yscale('log')  # set y-axis scale to logarithmic
    plt.savefig(os.path.join(ex_path, f'loss_ex{ex}.png'))
    plt.close()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Unet micromagnetics')
    parser.add_argument('--batch-size',   type=int,   default=100,    help='input batch size for training (default: 16)')
    parser.add_argument('--lr',           type=float, default=0.005,  help='learning rate (default: 0.002)')
    parser.add_argument('--epochs',       type=int,   default=1000,   help='number of epochs to train (default: 1000)')
    parser.add_argument('--pre_epochs',   type=int,   default=50,    help='number of pre_epochs to train' )
    parser.add_argument('--fz_epochs',    type=int,   default=10,   help='number of epochs to train (default: 1000)')
    parser.add_argument('--fz_nums',      type=int,   default=5,   help='number of epochs to train (default: 1000)')
    parser.add_argument('--inter_epochs', type=int,   default=10,   help='number of epochs to train (default: 1000)')
    
    parser.add_argument('--kc',        type=int,    default=16,     help='kernels of first layer (default: 16)')
    parser.add_argument('--inch',      type=int,    default=6,      help='input channels (default: 3)')
    parser.add_argument('--cornum',    type=int,    default=1000,   help='core number (default: 10000)')
    parser.add_argument('--ntest',     type=int,    default=10,     help='test number (default: 1000)')
    parser.add_argument('--ntrain',    type=int,    default=250,    help='train number (default: 10000)')

    parser.add_argument('--gpu',        type=int,   default=0,      help='GPU used (default: 0)')
    parser.add_argument('--ex',         type=float, default=1.0,    help='experiment (default: 0)')
    parser.add_argument('--dataug',     type=bool,  default=True,   help='data augmentation (default: False)')
    args = parser.parse_args()

    #working env
    device = torch.device("cuda:{}".format(args.gpu))
    torch.manual_seed(0)
    torch.cuda.manual_seed(0) 
    torch.backends.cudnn.benchmark = True
    
    # Model, optimizer, and data loaders initialization
    model = UNet(kc=args.kc, inc=args.inch, ouc=args.inch).to(device)

    optim = optimizer.Adam([
            {"params":model.parameters()},
    ], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=1, T_mult=2, eta_min=0.00001)
    #load data
    data_path11 = '../../../utils/Dataset/data_Hd32_Hext1000_mask'
    data_path12 = '../../../utils/Dataset/data_Hd32_Hext100_mask'
    data_path13 = '../../../utils/Dataset/data_Hd32_Hext0'

    data_path21 = '../../../utils/Dataset/data_Hd64_Hext1000_mask'
    data_path22 = '../../../utils/Dataset/data_Hd64_Hext100_mask'
    data_path23 = '../../../utils/Dataset/data_Hd64_Hext0'

    data_path31 = '../../../utils/Dataset/data_Hd128_Hext1000_mask'
    data_path32 = '../../../utils/Dataset/data_Hd128_Hext100_mask'
    data_path33 = '../../../utils/Dataset/data_Hd128_Hext0'

    data_path41 = '../../../utils/Dataset/data_Hd256_Hext1000_mask'
    data_path42 = '../../../utils/Dataset/data_Hd256_Hext100_mask'
    data_path43 = '../../../utils/Dataset/data_Hd256_Hext0'

    
    data_path1 = [data_path11, data_path12, data_path13]
    data_path2 = [data_path21, data_path22, data_path23]
    data_path3 = [data_path31, data_path32, data_path33]
    data_path4 = [data_path41, data_path42, data_path43]
    
    train_dataset1, test_dataset1 = dataset_prepare(data_path1, ntest=args.ntest, n128=args.ntest, ntrain=args.ntrain, cn=args.cornum)
    train_dataset2, test_dataset2 = dataset_prepare(data_path2, ntest=args.ntest, n128=args.ntest, ntrain=args.ntrain, cn=args.cornum)
    train_dataset3, test_dataset3 = dataset_prepare(data_path3, ntest=args.ntest, n128=args.ntest, ntrain=args.ntrain, cn=args.cornum)
    test_dataset4 = dataset_prepare(data_path4, ntest=0, n128=args.ntest, ntrain=0, cn=args.cornum, mode='eval256')

    bsz1=args.batch_size
    print('samples 1 2 3:',len(train_dataset1), len(train_dataset2), len(train_dataset3))
    bsz2=round(bsz1 / (len(train_dataset1) / len(train_dataset2)))
    bsz3=round(bsz1 / (len(train_dataset1) / len(train_dataset3)))
    print('batch size 1 2 3: ',bsz1, bsz2, bsz3,'\n')


    train_dataloader1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=bsz1, shuffle=True,  num_workers=0, drop_last=False)
    train_dataloader2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=bsz2, shuffle=True,  num_workers=0, drop_last=False)
    train_dataloader3 = torch.utils.data.DataLoader(dataset=train_dataset3, batch_size=bsz3, shuffle=True,  num_workers=0, drop_last=False)

    test_dataloader1  = torch.utils.data.DataLoader(dataset=test_dataset1,  batch_size=10, shuffle=True,  num_workers=0, drop_last=False)
    test_dataloader2  = torch.utils.data.DataLoader(dataset=test_dataset2,  batch_size=10, shuffle=True,  num_workers=0, drop_last=False)
    test_dataloader3  = torch.utils.data.DataLoader(dataset=test_dataset3,  batch_size=10, shuffle=True,  num_workers=0, drop_last=False)
    test_dataloader4  = torch.utils.data.DataLoader(dataset=test_dataset4,  batch_size=10, shuffle=True,  num_workers=0, drop_last=False)

    #experiment path
    ex_path='./ex{}_bsz{}_Ir{}_Unet_kc{}_inch{}'.format(
            args.ex, bsz1, args.lr, args.kc, args.inch
            )
    os.makedirs(ex_path, exist_ok=True)

    # Set up logging
    logging.basicConfig(filename=ex_path + '/training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    loss_train_list = []
    loss_test_list1 = []
    loss_test_list2 = []
    loss_test_list3 = []
    loss_test_list4 = []
    epoch_list = []

    epoch_num = args.epochs 


    best_loss = float('inf')

    for epoch in range(1,epoch_num+1):
        loss_train = train(epoch, model, optim, scheduler, train_dataloader1, train_dataloader2, train_dataloader3, ex_path)

        loss_train_list.append(loss_train)
        logging.info('epoch: {} loss: {:.2f}'.format(epoch, loss_train))
        #evaluate
        loss_test1, loss_test2, loss_test3, loss_test4, avg = eval(epoch, model, test_dataloader1, test_dataloader2, test_dataloader3, test_dataloader4, ex_path)
        logging.info('Evaluate loss32: {:.1f} loss64: {:.1f} loss128: {:.1f} / loss96: {:.1f} avg: {:.1f}'
                    .format(loss_test1, loss_test2, loss_test3, loss_test4, avg))
        print("\n")
        epoch_list.append(epoch)
        loss_test_list1.append(loss_test1)
        loss_test_list2.append(loss_test2)
        loss_test_list3.append(loss_test3)
        loss_test_list4.append(loss_test4)
        #save best model checkpoint
        loss_test_avg = (loss_test1+loss_test2+loss_test3+loss_test4)/4
        best_loss = save_best_model(model, epoch, loss_test_avg, best_loss, ex_path)

        draw_loss(epoch_list, loss_train_list, loss_test_list1, loss_test_list2, loss_test_list3, loss_test_list4, ex_path, args.ex)

        # reset best_loss
        if epoch == 800:
            best_loss = float('inf')
        if epoch == 900:
            best_loss = float('inf')
