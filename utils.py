import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np

import torch
from torch.autograd import Variable
import tqdm
import six

import os

import re
import itertools
from textwrap import wrap
import prepare_data

import io
import matplotlib
from sys import platform
if platform == "linux" or platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor

def timeStamped(fname='', fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Add a timestamp to your training run's name.
    """
    # http://stackoverflow.com/a/5215012/99379
    return datetime.now().strftime(fmt).format(fname=fname)


def get_run_dir_from_args(primary_train_result_dir, usecase, model, run_tag):
    train_type = usecase + '_' + model
    training_run_name = timeStamped(run_tag)
    training_run_dir = os.path.join(primary_train_result_dir, train_type, training_run_name)
    return training_run_dir


def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def get_confusion_matrix_img(numpy_cm, labels, font_size=20): # numpy-cm
    if len(labels) != numpy_cm.shape[0]:
        raise Exception("Length of labels %i does not match dimension of confusion matrix' height %i" % (len(labels), numpy_cm.shape[0]))
    # TODO Define figsize as a function of the longest string in labels
    total_sum = numpy_cm.sum()
    fig = plt.figure(figsize=(12, 12), dpi=60, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(numpy_cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=font_size)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=font_size, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=font_size)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=font_size, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(numpy_cm.shape[0]), range(numpy_cm.shape[1])):
        ax.text(j, i, "%.2f %%\n( %i )"%(numpy_cm[i, j]*100/total_sum, numpy_cm[i, j]) if numpy_cm[i, j] != 0 else '.', horizontalalignment="center"
                , fontsize=font_size, verticalalignment='center', color="black")

    # return fig
    #fig.show()

    # Convert to tensor
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    return image


def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer
          , n_epochs=None, fold=None, num_classes=None, callbacklist=None, tensorboard_writer=None):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)

    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        total_minibatch_count = state['total_minibatch_count']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        print('Building model from scratch')
        epoch = 1
        step = 0
        total_minibatch_count = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'total_minibatch_count': total_minibatch_count
    }, str(model_path))

    #report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    callbacklist.on_train_begin()
    for epoch in range(epoch, n_epochs + 1):
        #if epoch > 5: break
        callbacklist.on_epoch_begin(epoch)
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                #if i > 2: break
                callbacklist.on_batch_begin(i)
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                # TODO If the loss is significant, store a few output images that could be viewed from tensorflow
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])

                batch_logs = {
                    'loss': np.array(loss.data[0]),
                    'size': np.array(targets.shape[0])
                }

                batch_logs['batch'] = np.array(i)
                callbacklist.on_batch_end(i, batch_logs)

                if args.log_interval != 0 and total_minibatch_count % args.log_interval == 0:
                    mean_loss = np.mean(losses[-args.log_interval:])
                    tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                    write_event(log, step, loss=mean_loss)
                    # put all the logs in tensorboard
                    for name, value in six.iteritems(batch_logs):
                        tensorboard_writer.add_scalar(
                            name, value, global_step=total_minibatch_count)

                    # put all the parameters in tensorboard histograms
                    """"for name, param in model.named_parameters():
                        name = name.replace('.', '/')
                        tensorboard_writer.add_histogram(
                            name, param.data.cpu().numpy(), global_step=total_minibatch_count)
                        tensorboard_writer.add_histogram(
                            name + '/gradient',
                            param.grad.data.cpu().numpy(),
                            global_step=total_minibatch_count)"""
                total_minibatch_count += 1

            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics, confusion_matrix = validation(model, criterion, valid_loader, num_classes)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            # TODO Add early stopping

            for name, value in six.iteritems(valid_metrics):
                tensorboard_writer.add_scalar(
                    name, value, global_step=total_minibatch_count)
            if args.type == 'parts':
                _, _, labels = prepare_data.get_factor_mask_labels(args.type)

                cm_img = get_confusion_matrix_img(confusion_matrix, labels)  # Exclude background from labels
                tensorboard_writer.add_image("Confusion Matrix", cm_img, epoch)

            callbacklist.on_epoch_end(epoch, valid_metrics)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return