import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader

import argparse
from sklearn.metrics import jaccard_score

from models.model_multitask import MultiTaskModel
from smoke_plume_segmentation_dataset_endtoend import create_dataset

print('running on...', device)


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc


def eval_model(model, params, datadir, seglabeldir, reg_data):
    """Wrapper function for model evaluation.
    :param model: model instance
    :param params: parameters
    :param datadir: path to satellite images
    :param seglabeldir: path to segmentation labels
    :param reg_data: path to csv file for regression"""

    reg_data = pd.read_csv(os.path.join(reg_data, 'reg_co2_data.csv'))

    # create dataset
    data_val = create_dataset(
        datadir=datadir, seglabeldir=os.path.join(seglabeldir, 'validation/'),
        reg_data=reg_data, mult=1)

    val_dl = DataLoader(data_val, batch_size=params.bs)

    # define losses
    loss_r = nn.L1Loss()  # regression loss
    loss_c = nn.CrossEntropyLoss()  # classification loss
    loss_s = nn.BCEWithLogitsLoss()  # segmentation loss

    # evaluation
    model.eval()

    val_loss_total, val_bin_acc_total = 0, 0
    val_ious = []
    val_image_loss_total, val_gen_loss_total, val_bin_loss_total = 0, 0, 0

    progress = tqdm(enumerate(val_dl), desc="val Loss: ",
                    total=len(val_dl))

    for j, batch in progress:
        x = batch['img'].float().to(device)
        y = batch['fpt'].float().to(device)
        w = batch['weather'].float().to(device)
        e = batch['gen_output'].float().to(device)
        t = batch['type'].long().to(device)

        output, reg_output, logits = model(x, w)

        bin_acc = multi_acc(logits, t)
        val_bin_acc_total += bin_acc

        # derive loss
        loss_image = loss_s(output, y.unsqueeze(dim=1))
        loss_bin = loss_c(logits, t)
        loss_gen = loss_r(reg_output, e.unsqueeze(dim=1))

        val_image_loss_total += loss_image.item()
        val_bin_loss_total += loss_bin.item()
        val_gen_loss_total += loss_gen.item()

        loss_epoch = loss_image + loss_gen + loss_bin
        val_loss_total += loss_epoch.item()

        # derive binary segmentation map from prediction
        output_binary = np.zeros(output.shape)
        output_binary[output.cpu().detach().numpy() >= 0] = 1

        # derive IoU values
        for k in range(y.shape[0]):
            z = jaccard_score(y[k].flatten().cpu().detach().numpy(), output_binary[k][0].flatten())
            if (np.sum(output_binary[k][0]) != 0 and np.sum(y[k].cpu().detach().numpy()) != 0):
                val_ious.append(z)

    print(("total loss={:.3f}, segmentation loss={:.3f}, "
           "regression loss={:.3f}, classification loss={:.3f}, iou={:.3f}, classification acc={:.3f}").format(
               val_loss_total / (j + 1), val_image_loss_total / (j + 1), val_gen_loss_total / (j + 1),
               val_bin_loss_total / (j + 1), np.average(val_ious), val_bin_acc_total / (j + 1)))


def main():
    # setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', type=int, default=300,
                        help='Number of epochs')
    parser.add_argument('-bs', type=int, nargs='?',
                        default=32, help='Batch size')
    parser.add_argument('-lr', type=float,
                        nargs='?', default=0.7, help='Learning rate')
    parser.add_argument('-mo', type=float,
                        nargs='?', default=0.7, help='Momentum')
    parser.add_argument('-exp_name', type=str, default='',
                        help='Name of experiment')
    parser.add_argument('-lw', type=float, default=0.1,
                        help='Weight Loss')
    args = parser.parse_args()

    model = MultiTaskModel(n_channels=12, n_classes=1)
    model.to(device)

    checkpoint_path = 'path/to/model/checkpoint'
    model.load_state_dict(torch.load('{}'.format(checkpoint_path), map_location=torch.device('cpu')))
    model.to(device)

    # evaluate model
    eval_model(model, args, datadir='path/to/images', seglabeldir='path/to/seglabels', reg_data='path/to/csv')


if __name__ == '__main__':
    main()
