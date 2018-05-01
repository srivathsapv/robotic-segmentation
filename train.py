import argparse
import json
from pathlib import Path
from validation import validation_binary, validation_multi

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from models import UNet11, LinkNet34, UNet, UNet16
from loss import LossBinary, LossMulti
from dataset import RoboticsDataset
import utils
from tensorboardX import SummaryWriter
import callbacks
import os
import numpy as np
import datetime

from prepare_train_val import get_split

from transforms import (DualCompose,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        VerticalFlip)


def prepareDatasetAndLogging(args, train_dir):

    # kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Create the dataset, mnist or fasion_mnist
    def make_loader(file_names, shuffle=False, transform=None, problem_type='binary'):
        return DataLoader(
            dataset=RoboticsDataset(file_names, transform=transform, problem_type=problem_type),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split(args.fold, train_path=args.train_files_dir)

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    train_transform = DualCompose([
        HorizontalFlip(),
        VerticalFlip(),
        ImageOnly(Normalize())
    ])

    val_transform = DualCompose([
        ImageOnly(Normalize())
    ])

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform, problem_type=args.type)
    valid_loader = make_loader(val_file_names, transform=val_transform, problem_type=args.type)

    # Set up visualization and progress status update code
    callback_params = {'epochs': args.n_epochs,
                       'samples': len(train_loader) * args.batch_size,
                       'steps': len(train_loader),
                       'metrics': {'acc': np.array([]),
                                   'loss': np.array([]),
                                   'val_acc': np.array([]),
                                   'val_loss': np.array([])}}

    tensorboard_log_dir = os.path.join(str(train_dir), "tensorboard_logs")

    callbacklist = callbacks.CallbackList(
        [callbacks.BaseLogger(),
         callbacks.TQDMCallback(),
         callbacks.CSVLogger(filename=str(tensorboard_log_dir) + '/callback_logs.csv')])
    callbacklist.set_params(callback_params)
    tensorboard_writer = SummaryWriter(
        log_dir=tensorboard_log_dir)

    # show some image examples in tensorboard projector with inverted color
    # TODO Add sample images to tensorboard
    """images = valid_loader.dataset.test_data[:100].float()
    label = valid_loader.dataset.test_labels[:100]
    features = images.view(100, 784)
    tensorboard_writer.add_embedding(
        features,
        metadata=label,
        label_img=images.unsqueeze(1))
        """
    return tensorboard_writer, callbacklist, train_loader, valid_loader

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=1, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--train-result-dir', default='runs', help='checkpoint root')
    arg('--batch-size', type=int, default=1)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=8)
    arg('--type', type=str, default='binary', choices=['binary', 'parts', 'instruments'])
    arg('--model', type=str, default='UNet', choices=['UNet', 'UNet11', 'LinkNet34'])
    arg('--run-tag', type=str, default='', metavar='N',
                    help="""A tag for this training run, this
                            affects the directory so use underscores and not spaces.""")
    arg('--log_interval', type=int, default=100, metavar='I',
                    help="""how many batches to wait before logging detailed
                            training status, 0 means never log """)
    arg('--reuse-train-dir', type=str, default=None,
                    help="""If it is set, training would resume using results in the directory.""")

    arg('--train-files-dir', type=str, default="data/cropped_train",
                    help="""Location of directory containing instrument directories. .""")
    # TODO Add additional argument to reuse previous execution's parameters as default parameters. The previous execution's parameters can be obtained from params.json
    # TODO Optionally split reuse-train-dir into two arguments - train-dir and reuse (boolean)
    # TODO Currently after a training model is resumed, training would occur for {n_epochs - (last completed epoch no)}. Change this so that the new training job occurs for n_epochs unless a flag resume_from_prev_epoch is set

    args = parser.parse_args()

    # TODO If reuse_train_dir is set, add checks to ensure that the previous execution exists
    # TODO If reuse_train_dir is set, add checks to ensure that the previous execution's model and type match that of the current run (if explicitly provided)
    args.root = args.reuse_train_dir or \
                utils.get_run_dir_from_args(args.train_result_dir, args.type, args.model, args.run_tag)
    args.root = os.path.abspath(args.root)

    train_dir = Path(args.root)
    train_dir.mkdir(exist_ok=True, parents=True)

    if args.type == 'parts':
        num_classes = 4
    elif args.type == 'instruments':
        num_classes = 8
    else:
        num_classes = 1

    if args.model == 'UNet':
        model = UNet(num_classes=num_classes)
    elif args.model == 'UNet11':
        model = UNet11(num_classes=num_classes, pretrained='vgg')
    elif args.model == 'UNet16':
        model = UNet16(num_classes=num_classes, pretrained='vgg')
    elif args.model == 'LinkNet34':
        model = LinkNet34(num_classes=num_classes, pretrained=True)
    else:
        model = UNet(num_classes=num_classes, input_channels=3)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    if args.type == 'binary':
        loss = LossBinary(jaccard_weight=args.jaccard_weight)
    else:
        loss = LossMulti(num_classes=num_classes, jaccard_weight=args.jaccard_weight)

    cudnn.benchmark = True

    # TODO Modify this to include starting epoch. This would require moving epoch determination code from utils.train before this point.
    execution_time = utils.timeStamped()
    train_dir.joinpath('params_%s.json'%execution_time).write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    if args.type == 'binary':
        valid = validation_binary
    else:
        valid = validation_multi

    tensorboard_writer, callbacklist, train_loader, valid_loader = prepareDatasetAndLogging(args, train_dir)

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=args.fold,
        num_classes=num_classes,
        callbacklist=callbacklist,
        tensorboard_writer=tensorboard_writer
    )

    callbacklist.on_train_end()
    tensorboard_writer.close()


if __name__ == '__main__':
    main()
