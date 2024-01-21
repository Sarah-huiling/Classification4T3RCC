
import glob
from datetime import datetime

import pandas as pd
# #from setting import parse_opts
# from datasets.brains18 import BrainS18Dataset
# from model import generate_model
# !/usr/bin/env Python
# coding=utf-8
import torch
# import numpy as np
import xlrd
import xlwt
from torch import nn
from torch import optim
import random
import numpy as np
import logger
from data import MyDataset
# from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
# import time
# from utils.logger import log
# from scipy import ndimage
# import os
from torch.autograd import Variable
from modelLib.ResNet import ResNet50
from modelLib.MobileNetV2 import mobilenet_v2
from modelLib.vit3d import ViT3D
from modelLib.vit import pretrainedViT
from modelLib.pyramidnet import PyramidNet
from modelLib.mobile_vit import MobileViT
from sklearn import metrics as mt
import os


# os.environ['CUDA_LAUNCH_BLOCKING'] = "0, 1"
class BCELoss_class_weighted(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, input, target):
        # input = torch.clamp(input,min=1e-7,max=1-1e-7)  # 压缩到区间 [min,max]
        # input = torch.softmax(input)  # 压缩到区间 [min,max]
        bce = - (self.weight[1] * target * torch.log(input) + self.weight[0] * (1 - target) * torch.log(1 - input))
        return torch.mean(bce)


def train(alexnet_model, train_loader, epoch, train_dict, logger, criterion, use_gpu):
    alexnet_model.train()  # 训练模式，作用是启用 batch normalization 和 dropout
    losss = 0
    N_iter = 0
    for N_iter, batch in enumerate(train_loader):
        torch.cuda.empty_cache()
        if use_gpu:
            inputs = Variable(batch[0].cuda())
            labels = Variable(batch[1].cuda())
        else:
            inputs, labels = Variable(batch['0']), Variable(batch['1'])

        optimizer.zero_grad()  # reset gradient

        outputs = alexnet_model(inputs)
        # print(outputs, labels)
        if criterion == 'BCE_Loss':
            outputs = torch.softmax(outputs, dim=1)
        loss = eval(criterion)(outputs, labels)  #
        # loss = CrossEntropy_Loss(outputs, labels)  #
        # loss = BCEW_LogitsLoss(outputs, labels)
        # acc(outputs, labels)
        # backward反向传播，计算当前梯度
        loss.backward()  # 计算损失
        optimizer.step()  # 根据梯度更新网络/权重参数

        losss = losss + loss.item()
        # dice0, dice1, dice2, dice3 = dicev(outputs, labels)
        if (N_iter + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, N_iter, len(train_loader),
                100. * N_iter / len(train_loader), losss / (N_iter + 0.000001)))
    train_dict['loss'].append(losss / (N_iter + 0.000001))
    logger.scalar_summary('train_loss', losss / (N_iter + 0.000001), epoch)


def val_test(alexnet_model, val_loader, criterion):
    val_path = val_loader.dataset.image_files
    alexnet_model.eval()  # 评估(推断)的模式
    val_loss = 0
    with torch.no_grad():
        p = []
        g = []
        for N_iter, batch in enumerate(val_loader):
            torch.cuda.empty_cache()
            if use_gpu:
                inputs = Variable(batch[0].cuda())
                labels = Variable(batch[1].cuda())
                # infos = Variable(batch[2].cuda())
            else:
                inputs, labels, infos = Variable(batch['0']), Variable(batch['1']), Variable(batch['2'])

            outputs = alexnet_model(inputs)
            outputs = torch.softmax(outputs, dim=1)

            if criterion == 'BCE_Loss':
                outputs = torch.softmax(outputs, dim=1)
            loss = eval(criterion)(outputs, labels)
            labels = labels.cpu().numpy()
            for x, y in zip(outputs, labels):
                p.append(x)
                g.append(y)
            val_loss += loss.item()
        auc, gt2, pr_pob, pr_neg, pr_pos = ODIR_Metrics(np.array(p), np.array(g))
    val_loss /= len(val_loader)
    print('\nAverage loss: {:.6f},auc: {:.6f}\n'.format(val_loss, auc))
    return auc, val_loss, gt2, pr_pob, pr_neg, pr_pos, val_path


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = 0.001 * (0.1 ** (epoch // 25))
    # lr = init_lr * (0.1 ** (epoch // 20))
    lr = init_lr * (0.95 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def ODIR_Metrics(pred, target):
    th = 0.5
    gt = target.flatten()
    pr = pred.flatten()

    gt1 = gt[0::2]
    pr_neg = pr[0::2]  # pr_neg 预测为阴性的概率
    gt2 = gt[1::2]
    pr_pos = pr[1::2]  # pr_pos 预测为阳性的概率

    gt_prePob = []
    for i in range(len(gt2)):
        if gt2[i] == 1:
            gt_prePob.append(pr_pos[i])
        if gt2[i] == 0:
            gt_prePob.append(pr_neg[i])
    preLabel = np.zeros(len(gt2))
    preLabel[pr_pos > th] = 1

    print('=' * 20)
    print('gt2.shape', gt2.shape)
    print('pr2.shape', pr_pos.shape)
    # print('pr_pos.shape', len(pr_pos))
    # fpr, tpr, thresholds = mt.roc_curve(gt2, pr_pos, pos_label=1.0)  # 阳性的概率：以1类作为阳性，则输入预测为1的概率
    # roc_auc2 = mt.auc(fpr, tpr)
    kappa = mt.cohen_kappa_score(gt, pr > th)
    print("1：auc值,", mt.roc_auc_score(gt1, pr_neg), 'acc:', mt.accuracy_score(gt1, pr_neg > th))
    print("2：auc值,", mt.roc_auc_score(gt2, pr_pos), 'acc:', mt.accuracy_score(gt2, pr_pos > th))
    # f1 = mt.f1_score(gt, pr > th, average='micro')
    roc_auc = mt.roc_auc_score(gt2, pr_pos)
    return roc_auc, gt2, gt_prePob, pr_neg, pr_pos


def load_label(excelpath, dataformat, label_column=1):
    PIDs = []
    label = []
    # print(os.path.splitext(excelpath)[-1])
    if dataformat == '.csv':
        data = pd.read_csv(excelpath)
        PIDs = list(data.values[:, 0])
        label = list(data.values[:, label_column])
    if dataformat == '.xls' or dataformat == '.xlsx':
        reads = xlrd.open_workbook(excelpath)
        for row in range(1, reads.sheet_by_index(0).nrows):
            PIDs.append(reads.sheet_by_index(0).cell(row, 0).value)
            label.append(reads.sheet_by_index(0).cell(row, label_column).value)
    return PIDs, label


class WeightedMultilabel(torch.nn.Module):

    def __init__(self, weights: torch.Tensor):
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.weights = weights.unsqueeze()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets) * self.weights


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


if __name__ == "__main__":
    # batch_size = 128
    batch_size = 64
    epochs = 150
    lr = 0.001
    momentum = 0.95
    w_decay = 1e-6
    # step_size = 20
    gamma = 0.5
    n_class = 2
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    a = []
    data_path = '/media/zhl/ResearchData/20230811华西肾癌/T3RCC/DL-classification/npy_internal'
    data_pathzq = '/media/zhl/ResearchData/20230811华西肾癌/T3RCC/DL-classification/npy_internal_zq'
    AllPID_label_path = '/media/zhl/ResearchData/20230811华西肾癌/T3RCC/DL-classification/label_internal.xls'
    Root_SavePath = '/media/zhl/ResearchData/20230811华西肾癌/T3RCC/DL-classification/ClassificationResult/internalValidation/label_any'
    modelName = 'PyramidNet'
    curr_datetime = datetime.now()
    DLResultPath = check_dir(os.path.join(Root_SavePath, modelName + str(curr_datetime).split(':')[0]
                                          + ':' + str(curr_datetime).split(':')[1] + '_CrossEntropy_Loss_label_any'))
    trainvalpath = check_dir(DLResultPath + '/trainValPath')
    modelPath = check_dir(DLResultPath + '/model')
    probPath = check_dir(DLResultPath + '/prob')
    aucExcelPath = check_dir(DLResultPath + '/MetricsExcel')
    LogPath = check_dir(DLResultPath + '/Log')
    # # 读取 ImagPath和label
    data_npy, label = load_label(AllPID_label_path, dataformat=os.path.splitext(AllPID_label_path)[-1], label_column=3)

    t_path = []
    npy_path_pos = []
    npy_path_neg = []
    for i in range(len(data_npy)):
        t_path.append(os.path.join(data_path, str(data_npy[i]) + '.npy'))
        if label[i] == 0:
            npy_path_neg.append(os.path.join(data_path, str(data_npy[i]) + '.npy'))
        if label[i] == 1:
            npy_path_pos.append(os.path.join(data_path, str(data_npy[i]) + '.npy'))
    random.shuffle(npy_path_neg)
    random.shuffle(npy_path_pos)
    # leng = len(t_path)
    test_path = []
    folds = 5  # 做7:3
    for fold in range(0, folds):
        # test_path = t_path[fold * int(leng/folds):(fold+1) * int(leng/folds)]
        npy_path_neg_test = npy_path_neg[
                            fold * int(len(npy_path_neg) / folds):(fold + 1) * int(len(npy_path_neg) / folds)]
        npy_path_pos_test = npy_path_pos[
                            fold * int(len(npy_path_pos) / folds):(fold + 1) * int(len(npy_path_pos) / folds)]
        test_path = npy_path_neg_test + npy_path_pos_test
        # print(test_path)
        # # 保存test path
        random.shuffle(test_path)
        path = trainvalpath + '/test_fold' + str(fold) + '.xls'
        f = xlwt.Workbook()
        sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=False)  # 创建sheet
        for i in range(len(test_path)):
            sheet1.write(i, 0, str(test_path[i]))
        f.save(path)

        train_path = list(set(t_path).difference(set(test_path)))  # t_path中有而test_path中没有
        train_path_merge = []
        for train_path1 in train_path:
            train_path_temp = train_path1.split('.')[0]
            PID_Name = os.path.split(train_path_temp)[-1]
            train_path2 = os.path.join(data_pathzq, PID_Name + '_HFV.npy')
            # train_path3 = os.path.join(data_pathzq, PID_Name + '_scale07.npy')
            train_path4 = os.path.join(data_pathzq, PID_Name + '_HF.npy')
            train_path5 = os.path.join(data_pathzq, PID_Name + '_V.npy')
            # train_path6 = os.path.join(data_pathzq, PID_Name + '_HFV.npy')
            train_path1234 = [train_path1] + [train_path2] + [train_path4] + [train_path5]
            train_path_merge += train_path1234
        # 保存train_path加入增强的
        train_path = train_path_merge
        random.shuffle(train_path)
        path = trainvalpath + '/train_fold' + str(fold) + '.xls'
        f = xlwt.Workbook()
        sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=False)  # 创建sheet
        for i in range(len(train_path)):
            sheet1.write(i, 0, str(train_path[i]))
        f.save(path)

        train_da = MyDataset(train_path, transform=False)
        test = MyDataset(test_path, transform=False)
        # val = MyDataset(val_path, transform=False)
        train_loader = DataLoader(train_da, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)
        # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2)

        print('model load...')
        model_dir = modelPath
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if modelName == 'mobilenet':
            # model = mobilenet_v2(in_c=3, num_classes=2, pretrained=False, dropoutP=0.8)
            model = mobilenet_v2(in_c=99, num_classes=2, pretrained=False)
            # model = mobilenet_v2(in_c=3, num_classes=2, pretrained=False, input_size=224)
        if modelName == 'MobileViT':
            model = MobileViT(num_classes=2)
        if modelName == 'PyramidNet':
            model = PyramidNet(in_c=99, dataset='imagenet', depth=50, alpha=100, num_classes=2)
        if modelName == 'Resnet':
            model = ResNet50(in_c=99, num_classes=2)
        if modelName == 'vit3D':
            model = ViT3D(
                # image_size=(320, 320, 33),
                image_size=168,
                patch_size=32,
                num_classes=2,
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
            )
        # if modelName == 'vgg':
        #     model = vgg16_bn(num_classes=2)
        if modelName == 'vit':
            preModel = '/media/zhl/ProgramCode/DL_Classification/Classification_ImgClinic/vit_base_patch32_224_in21k.pth'
            # model = VisionTransformer(in_c=3, num_classes=2, patch_size=32, img_size=512, drop_ratio=0.8,
            # attn_drop_ratio=0.8, drop_path_ratio=0.8)
            model = pretrainedViT(pretrained=False, in_c=99, num_classes=2, patch_size=32, img_size=168
                                  , model_dir=preModel
                                  , drop_ratio=0.3)
            # , attn_drop_ratio=0.5
            # , drop_path_ratio=0.2)

        if use_gpu:
            alexnet_model = model.cuda()
            alexnet_model = nn.DataParallel(alexnet_model, device_ids=num_gpu)
            # alexnet_model = nn.parallel.DistributedDataParallel(alexnet_model, device_ids=num_gpu,
            #                                                     broadcast_buffers=False,
            #                                                     find_unused_parameters=True)
        else:
            alexnet_model = model

        weight = torch.FloatTensor([0.1, 9]).cuda() # 0.67
        CrossEntropy_Loss = nn.CrossEntropyLoss(weight=weight)

        criterion = 'CrossEntropy_Loss'
        optimizer = optim.Adam(alexnet_model.parameters(), lr=lr, betas=(0.9, 0.99))
        # optimizer = optim.SGD(alexnet_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
        # create dir for score
        score_dir = os.path.join(model_dir, 'scores')
        if not os.path.exists(score_dir):
            os.makedirs(score_dir)
        train_dict = {'loss': []}
        val_dict = {'loss': [], 'auc': []}
        logger1 = logger.Logger(LogPath)
        best_loss = 0
        Results = []  # 创建二位空矩阵
        # epochs = 39 + 1
        for i in range(7):
            Results.append([])
            for j in range(epochs + 1):
                Results[i].append([])

        for epoch in range(1, epochs + 1):
            # if epoch == 2:
            #     break
            # print(val_dict['loss'][0])
            train(alexnet_model, train_loader, epoch, train_dict, logger1, criterion, use_gpu)
            print("------------------------fold", fold, '------------------------------')
            print("------------------------epoch", epoch, '------------------------------')
            print("------------------------", 'auc_train', '------------------------------')
            auc_train, loss_train, gt_train, pr_train, pr_train0, pr_train1, train_path = val_test(alexnet_model,
                                                                                                   train_loader,
                                                                                                   criterion)
            # print("------------------------", 'auc_val', '------------------------------')
            # auc_val, loss_val = val_test(alexnet_model,  val_loader)
            print("------------------------", 'auc_test', '------------------------------')
            auc_test, loss_test, gt_test, pr_test, pr_test0, pr_test1, test_path = val_test(alexnet_model,
                                                                                            test_loader,
                                                                                            criterion)
            adjust_learning_rate(optimizer, epoch, lr)

            Results[0][0] = 'epoch'
            Results[1][0] = 'auc_train'
            Results[2][0] = 'loss_train'
            # Results[3][0] = 'auc_val'
            # Results[4][0] = 'loss_val'
            Results[5][0] = 'auc_test'
            Results[6][0] = 'loss_test'

            Results[0][epoch] = epoch
            Results[1][epoch] = auc_train
            Results[2][epoch] = loss_train
            # Results[3][epoch] = auc_val
            # Results[4][epoch] = loss_val
            Results[5][epoch] = auc_test
            Results[6][epoch] = loss_test

            if auc_train > 0.8 and auc_test > 0.8:
                model_path = os.path.join(model_dir, str(auc_test)[:4] + '_model.pth')
                torch.save(alexnet_model, model_path)

        #
        # 结果保存到excel
        # 将数据写入第 i 行，第 j 列
        f = xlwt.Workbook()
        sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=False)  # 创建sheet
        for i in range(7):
            sheet1.write(i, 0, str(Results[i][0]))
            # for j in range(np.size(datas)):
            for j in range(epoch):
                sheet1.write(i, j + 1, Results[i][j + 1])  # 将data[j] 写入第i行j列excel2003最大列为256
        path = aucExcelPath + '/Results' + '_AUC_fold' + str(fold) + '.xls'
        f.save(path)

    print('finished')

    # filename = 'Results.json'
    # with open(filename, 'w') as file_obj:
    #     json.dump(Results, file_obj)
