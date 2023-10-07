import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] ='1'
import platform
import pathlib
from torch.optim import lr_scheduler
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
import torch.backends.cudnn as cudnn
from model.bra_unet import BRAUnet
from config import get_config
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/Psfh', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Psfh', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Psfh', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--output_dir', default='./output/Psfh', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=51000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
args.root_path1 = os.path.join(args.root_path, "train_npz")
args.root_path2 = os.path.join(args.root_path, "test_vol_h5")
config = get_config(args)


def trainer_Psfh(args, model, snapshot_path):
    from datasets.dataset_Psfh import Psfh_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Psfh_dataset(base_dir=args.root_path1, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print(db_train)
    print("The length of train set is: {}".format(len(db_train)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    db_test = Psfh_dataset(base_dir=args.root_path2, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    save_best_path = None
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            p1 = model(image_batch)
            outputs = p1
            loss_ce = ce_loss(p1, label_batch[:].long())
            loss_dice = dice_loss(p1, label_batch, softmax=True)
            loss = 0.6 * loss_dice + 0.4 * loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            print('iteration %d : loss : %f,  loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))


            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 1000 == 0:
                print('Current learning rate:', optimizer.param_groups[0]['lr'])
                model.eval()
                metric_list = 0.0
                total_loss = 0.0
                for i_batch, sampled_batch in enumerate(testloader):
                    image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
                    metric_i, prediction, loss = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],test_save_path=None, case=case_name,z_spacing=1)
                    total_loss += loss
                    metric_list += np.array(metric_i)
                val_loss = total_loss / len(db_test)
                print('Validation Loss:', val_loss)
                metric_list = metric_list / len(db_test)
                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                if performance > best_performance:
                    best_performance = performance
                    if save_best_path and os.path.exists(save_best_path):
                        os.remove(save_best_path)
                    save_best_path = os.path.join(snapshot_path, 'best_model.pth'.format(performance))
                    torch.save(model.state_dict(), save_best_path)
                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f  ' % (iter_num, performance, mean_hd95))
            model.train()
        save_interval = 10  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 10) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Psfh': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Psfh',
            'num_classes': 3,
        },

    }
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = BRAUnet(img_size=256, num_classes=3, n_win=8, embed_dim=96).cuda()
    trainer = {'Psfh': trainer_Psfh(args, net, args.output_dir),}