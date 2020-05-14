import argparse
try:
    import moxing as mox
except:
    print('not use moxing')

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# import test  # import test.py to get mAP after each epoch
import codecs
from models import *
from utils.datasets import *
from utils.utils import *
from obs import *
import test

# mixed_precision = True
# try:  # Mixed precision training https://github.com/NVIDIA/apex
#     from apex import amp
# except:
mixed_precision = False  # not installed

# wdir = 'weights' + os.sep  # weights dir
# last = wdir + 'last.pt'
# best = wdir + 'best.pt'
# results_file = 'results.txt'
from tqdm import tqdm

# Hyperparameters https://github.com/ultralytics/yolov3/issues/310
# required 设置为True则意味着该参数为必须输入项。
parser = argparse.ArgumentParser(description="yolov3 pytorch")
parser.add_argument('--epochs', type=int, default=300)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
# parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
parser.add_argument('--img-size', nargs='+', type=int, default=[416], help='train and test image-sizes')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
parser.add_argument('--notest', action='store_true', help='only test final epoch')
parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
# parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--adam', action='store_true', help='use adam optimizer')
parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')


hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.001,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v


def train():
    """
    log_dir: 存储地址
    root_dir: VOC2007文件夹在其中的地址
    train_classes: 标签地址
    cfg: cfg文件地址
    img_size: 图像切割的大小
    epochs: 一共多少个epoch
    batch_size: batch_size大小
    weights: 预训练权重地址
    train_path: 训练集地址
    test_path: 测试集地址（这里我没有使用，原版作者是用了test，我是直接训练集合loss，这里没有使用）。
    """
    # 相关参数地址
    log_dir = "output"  # 输出保存地址地址
    root_dir = "trainval"  # VOC2007文件夹在其中的地址
    train_classes = os.path.join(root_dir, 'ImageSets/trainval.txt')  # 标签地址
    with codecs.open(train_classes, 'r', 'utf-8') as f:  # 读取标签
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    cfg = os.path.join(root_dir, "cfg/yolov3.cfg")  # 获取得到cfg文件的地址
    img_size, img_size_test = opt.img_size if len(opt.img_size) == 2 else opt.img_size * 2  # train, test sizes
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # 优化之前要累计的批次 ：effective bs = batch_size * accumulate = 16 * 4 = 64
    # weights = os.path.join(os.path.dirname(__file__), "yolo/yolov3.weights")  # 预训练权重地址, 本地训练使用
    weights = os.path.join(root_dir, 'weights/yolov3.weights')  # 预训练权重地址（华为）

    # Initialize
    init_seeds()  # 生成随机数

    # Configure run
    # train_path = os.path.join(os.path.dirname(__file__), 'trainval/VOC2007/hanxu.txt')  # 训练集地址
    train_path = os.path.join(root_dir, "ImageSets/trainval.txt")
    test_path = os.path.join(root_dir, "ImageSets/test.txt")
    test_data = [test_path, classes]

    nc = 44  # number of classes  类别数量

    # print("cfg:", cfg)
    # print("weights:", weights)
    # print("train_path", train_path)

    # Initialize model
    model = Darknet(cfg).to(device)

    # 优化器设置
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    # 选择优化器
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    # 设置epoch过程相关参数
    start_epoch = 0
    best_fitness = 0.0

    # 解析并加载存储在“权重”中的权重
    load_darknet_weights(model, weights)

    # Mixed precision training https://github.com/NVIDIA/apex 是否使用混合精度计算
    # if mixed_precision:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    # pytorch学习率调整，这里是自定义调整（https://blog.csdn.net/zisuina_2/article/details/103258573）
    # optimizer:优化器,,lr_lambda: 为optimizer.param_groups中的每个组计算一个乘法因子
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine https://arxiv.org/pdf/1812.01187.pdf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)

    # Initialize distributed training
    # 分布式训练，自行选择
    # if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
    #     dist.init_process_group(backend='nccl',  # 'distributed backend'
    #                             init_method='tcp://127.0.0.1:9999',  # distributed training init method
    #                             world_size=1,  # number of nodes for distributed training
    #                             rank=0)  # distributed training node rank
    #     model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    #     model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(root_dir, train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=False,  # rectangular training
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(root_dir, test_path, img_size_test, batch_size,
                                                                 hyp=hyp,
                                                                 rect=True,
                                                                 cache_images=opt.cache_images,
                                                                 single_cls=opt.single_cls),
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 0.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights

    # Model EMA
    ema = torch_utils.ModelEMA(model)

    # 开始训练
    nb = len(dataloader)  # number of batches
    prebias = start_epoch == 0
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # 对lr和权重的在训练过程中的特殊处理
        # Prebias
        if prebias:
            ne = 3  # number of prebias epochs
            ps = 0.1, 0.9  # prebias settings (lr=0.1, momentum=0.9)
            if epoch == ne:
                ps = hyp['lr0'], hyp['momentum']  # normal training settings
                model.gr = 1.0  # giou loss ratio (obj_loss = giou)
                print_model_biases(model)
                prebias = False

            # Bias optimizer settings
            optimizer.param_groups[2]['lr'] = ps[0]
            if optimizer.param_groups[2].get('momentum') is not None:  # for SGD but not Adam
                optimizer.param_groups[2]['momentum'] = ps[1]

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        loss_sum = []
        mloss = torch.zeros(4).to(device)  # mean losses
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)批次batches
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # 超参数特殊处理
            # Hyperparameter Burn-in
            n_burn = 300  # number of burn-in batches
            if ni <= n_burn:
                g = (ni / n_burn) ** 2  # gain
                for x in model.named_modules():  # initial stats may be poor, wait to track
                    if x[0].endswith('BatchNorm2d'):
                        x[1].track_running_stats = ni == n_burn
                for x in optimizer.param_groups:
                    x['lr'] = x['initial_lr'] * lf(epoch) * g  # gain rises from 0 - 1
                    if 'momentum' in x:
                        x['momentum'] = hyp['momentum'] * g


            # Run model
            """
            将取出的数据imgs传入model中，model就是yolov3的darknet，
            它有3个yolo层，每个yolo层都会输出一个特征映射图（dimention如(bs, 3, 13, 13, 49)）49=44+5 44是类别数量
            bs=batch_size,3指每一个像素点存在3种anchor，13*13是它的维度，pred = (xywh)+scores+classes。
            """
            pred = model(imgs)

            # Compute loss
            # targets = [image, class, x, y, w, h]
            loss, loss_items = compute_loss(pred, targets, model)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / batch_size * accumulate  # 64 = batch_size * accumulate
            loss_e = float(loss.cpu().detach().numpy())  # 转成numpy（）
            loss_sum.append(loss_e)

            # Compute gradient 是否使用混合精度计算
            # if mixed_precision:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     loss.backward()
            loss.backward()

            # Optimize accumulated gradient
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs

        results, maps = test.test(cfg,
                                  test_data,
                                  batch_size=batch_size,
                                  img_size=img_size_test,
                                  model=ema.ema,
                                  conf_thres=0.001 if final_epoch else 0.01,  # 0.001 for best mAP, 0.01 for speed
                                  iou_thres=0.6,
                                  save_json=False,
                                  single_cls=opt.single_cls,
                                  dataloader=testloader)
        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi
        print("Epoch: ", epoch, "------|| ", "Loss: ", list(mloss)[-1].item(), "------|| ", "map: ", list(maps)[2], end="\n")

        # save model
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, "epoch{}.pt".format(epoch+1)))
            w_path = "epoch{}.pt".format(epoch+1)

        # end epoch ---------------------------------------------------------------------------------------------------

    torch.save(model.state_dict(), os.path.join(log_dir, "last.pt".format(epoch + 1)))
    w_path = "last.pt".format(epoch + 1)
    # end training





if __name__ == '__main__':
    opt, unknown = parser.parse_known_args()
    # device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    # if device.type == 'cpu':  # 是否使用混合精度计算
    #     mixed_precision = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("use:", device)
    print("begin to train")
    train()
