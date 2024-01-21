"""


"""

# !/usr/bin/env python
# coding=utf-8
import random
import io
from sklearn.metrics import roc_auc_score, average_precision_score
import openpyxl
import torch
import glob
import numpy as np
from torch import nn
from Evaluate4MulLabel import compute_mAP, auROC
# from huaxi_cf_dataset import MyDataset
# from huaxi_dataset import MyDataset
# from data_WSI_tumor1vs2 import MyDataset
# from data_WSI import MyDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn import metrics as mt
# from skimage import io, color
from PIL import Image, ImageDraw, ImageFont
import os
import pylab as plt
import xlrd
import xlwt
import pandas as pd
# import csv
# import codecs
from sklearn.metrics import confusion_matrix
# from data_img4binary import MyDataset
from data import MyDataset


os.environ['CUDA_LAUNCH_BLOCKING'] = "0, 1"


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def draw_auc(fpr, tpr, name):
    roc_auc = mt.auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    tile = name + ' ROC'
    plt.title(tile)
    plt.plot(fpr, tpr, 'b', label=name + ' = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def drawCM(matrix, savname):
    # Display different color for different elements
    lines, cols = matrix.shape
    sumline = matrix.sum(axis=1).reshape(lines, 1)
    ratiomat = matrix / sumline
    toplot0 = 1 - ratiomat
    toplot = toplot0.repeat(50).reshape(lines, -1).repeat(50, axis=0)
    # io.imsave(savname, color.gray2rgb(toplot))
    # Draw values on every block
    image = Image.open(savname)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(os.path.join(os.getcwd(), "draw/ARIAL.TTF"), 15)
    for i in range(lines):
        for j in range(cols):
            dig = str(matrix[i, j])
            if i == j:
                filled = (255, 181, 197)
            else:
                filled = (46, 139, 87)
            draw.text((50 * j + 10, 50 * i + 10), dig, font=font, fill=filled)
    image.save(savname, 'jpeg')


def va(gt2, pr2, th):
    value_0 = {'tp': 0, 'tn': 0, 'fn': 0, 'fp': 0}
    for i in range(len(gt2)):
        if gt2[i] == 1 and pr2[i] >= th:
            value_0['tp'] = value_0['tp'] + 1  # 真正例
        if gt2[i] == 0 and pr2[i] >= th:
            value_0['fp'] = value_0['fp'] + 1  # 假正例
        if gt2[i] == 0 and pr2[i] < th:
            value_0['tn'] = value_0['tn'] + 1  # 真负例
        if gt2[i] == 1 and pr2[i] < th:
            value_0['fn'] = value_0['fn'] + 1  # 假负例
    return value_0


def ODIR_Metrics(pred, target):
    # corrected
    th = 0.5
    gt = target.flatten()
    pr = pred.flatten()

    gt1 = gt[0::2]
    pr_neg = pr[0::2]  # pr_neg 预测为阴性的概率
    gt2 = gt[1::2]
    pr_pos = pr[1::2]  # pr_pos 预测为阳性的概率

    gt_prePob = []  # 预测gt的prob
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
    kappa = mt.cohen_kappa_score(gt, pr > th)
    # print("1：auc值,", mt.roc_auc_score(gt1, pr_neg), 'acc:', mt.accuracy_score(gt1, pr_neg > th))
    # # f1 = mt.f1_score(gt, pr > th, average='micro')
    # roc_auc = mt.roc_auc_score(gt2, pr_pos)
    fpr, tpr, thresholds = mt.roc_curve(gt2, pr_pos, pos_label=1.0)
    roc_auc = mt.auc(fpr, tpr)
    print("auc: ,", roc_auc)
    return roc_auc, gt2, gt_prePob, pr_neg, pr_pos


def val_test(alexnet_model, val_loader, criterion):
    val_path = val_loader.dataset.image_files
    alexnet_model.eval()
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

            if criterion == 'BCE_Loss':
                outputs = torch.softmax(outputs, dim=1)  # BCEloss,sigmoid归一化到（0,1）后再送入
            loss = eval(criterion)
            loss = loss(outputs, labels)
            outputs = outputs.data.cpu().numpy()
            labels = labels.cpu().numpy()
            for x, y in zip(outputs, labels):
                p.append(x)
                g.append(y)
            val_loss += loss.item()
        auc, gt, pr_pob, pr_neg, pr_pos = ODIR_Metrics(np.array(p), np.array(g))
    val_loss /= len(val_loader)
    print('\nAverage loss: {:.6f},auc: {:.6f}\n'.format(val_loss, auc))
    return auc, val_loss, gt, pr_pob, pr_neg, pr_pos, val_path


def load_label(excel_path, dataformat, label_orNot=False, label_index=1):
    if dataformat == '.csv':
        data = pd.read_csv(excel_path)
    else:
        data = pd.read_excel(excel_path)
    label = []
    PIDs = list(data.values[:, 0])
    if label_orNot:
        label = list(data.values[:, label_index])
    return PIDs, label


if __name__ == "__main__":
    batch_size = 64  # 8  32
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    # trained model
    AllPID_label_path = '/media/zhl/ResearchData/20230811华西肾癌/T3RCC/DL-classification/label_external.xls'
    root_data_path = '/media/zhl/ResearchData/20230811华西肾癌/T3RCC/DL-classification/npy_external'
    root_savePath = '/media/zhl/ResearchData/20230811华西肾癌/T3RCC/DL-classification/ClasResult/externalVal/label_any'
    for i in range(1):
        model_dir = '/media/zhl/ResearchData/20230811华西肾癌/T3RCC/DL-classification/ClasResult/internalValidation/label_any/vit2023-09-02 16:56_BCE_Loss_label_any/model/0.82_train_fold4vit' # 0.857
        # external validation
        testPIDs, _ = load_label(AllPID_label_path, dataformat=os.path.splitext(AllPID_label_path)[-1],
                                 label_orNot=True, label_index=3)
        test_path = [os.path.join(root_data_path, str(i)+'.npy') for i in testPIDs]

        test_dataset = MyDataset(test_path)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

        print('model load...')
        model = torch.load(model_dir)
        print(model)
        BCE_Loss = nn.BCELoss()
        criterion = 'BCE_Loss'
        print('----------------------test value----------------------')
        auc_test, loss_test, gt_test, pr_test, pr_test0, pr_test1, test_pat = val_test(model, test_loader, criterion)
        print(f'auc_test:{auc_test}')
        print(f'loss_test:{loss_test}')

        # 指定文件的路径
        path = os.path.join(root_savePath, str(auc_test)[:4] + '_test.xls')
        f = xlwt.Workbook()
        sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=False)  # 创建sheet
        sheet1.write(0, 0, 'test_path')
        sheet1.write(0, 1, 'gt_test')
        sheet1.write(0, 2, 'pr_test')
        sheet1.write(0, 3, 'pr_test0')
        sheet1.write(0, 4, 'pr_test1')
        for i in range(len(gt_test)):
            sheet1.write(i + 1, 0, str(test_path[i]))
            sheet1.write(i + 1, 1, int(gt_test[i]))
            sheet1.write(i + 1, 2, float(pr_test[i]))
            sheet1.write(i + 1, 3, float(pr_test0[i]))
            sheet1.write(i + 1, 4, float(pr_test1[i]))
        f.save(path)
    print('finish')
