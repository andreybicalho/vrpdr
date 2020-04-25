import torch
import torchvision

from itertools import product
from collections import namedtuple
from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import time
import json
import logging

def info():
    logging.info(torch.__version__)
    logging.info(torch.cuda.is_available())
    logging.info(torch.version.cuda)
    logging.info(torch.cuda.current_device())
    logging.info(torch.cuda.get_device_name(0))
    
def get_runs_params(params):        
    Run = namedtuple('Run', params.keys())
    runs = []
    for v in product(*params.values()):
        runs.append(Run(*v))
    return runs

class TrainingManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.model = None
        self.loader = None
        self.tb = None

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def begin_run(self, run, model, loader):
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.model = model
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        # FIXME(andrey): adding model to tensorboard is crashing when model is running on GPU
        #self.tb.add_graph(self.model, images)

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

        logging.info(f'Starting epoch: {self.epoch_count}')

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        logging.info(f'\nFinished epoch {self.epoch_count} in {epoch_duration}s - Accuracy: {accuracy} - Loss: {loss}')

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.model.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        torch.save(self.model.state_dict(), f'{self.run_count}_{self.epoch_count}_model.pt')

    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_num_corret(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)    

    def save(self, filename):
        """
            Save results and model.
        """
        # save csv
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns'
        ).to_csv(f'{filename}.csv')

        # save json
        with open(f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

        torch.save(self.model.state_dict(), f'{filename}_model.pt')