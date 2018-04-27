import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np

import torch
from torch.autograd import Variable
import tqdm
import six

def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer
          , n_epochs=None, fold=None, num_classes=None, callbacklist = None, tensorboard_writer = None):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)

    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        print('Building model from scratch')
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    #report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    callbacklist.on_train_begin()
    total_minibatch_count = 0
    for epoch in range(epoch, n_epochs + 1):
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
                callbacklist.on_batch_begin(i)
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
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
                    for name, param in model.named_parameters():
                        name = name.replace('.', '/')
                        tensorboard_writer.add_histogram(
                            name, param.data.cpu().numpy(), global_step=total_minibatch_count)
                        tensorboard_writer.add_histogram(
                            name + '/gradient',
                            param.grad.data.cpu().numpy(),
                            global_step=total_minibatch_count)
                total_minibatch_count += 1

            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, num_classes)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)

            for name, value in six.iteritems(valid_metrics):
                tensorboard_writer.add_scalar(
                    name, value, global_step=total_minibatch_count)
            callbacklist.on_epoch_end(epoch, valid_metrics)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
