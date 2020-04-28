import torch
import logging
import numpy as np

from EMNISTNet.models import EMNISTNet

class OCR():
    """
    Optical Character Recognition (OCR) aims to recognize characters from images.
    """

    def __init__(self, model_filename="config/emnist_model.pt", num_classes=47, use_cuda=False, debug=False):

        if(debug):
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

        self._debug = debug
        # indexed classes labels
        self._groundtruth = ['0','1','2','3','4','5','6','7','8','9', # 10 classes (MNIST)
                             'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z', # 36 classes (custom dataset)
                             'a','b','d','e','f','g','h','n','q','r','t'] # 47 classes (EMNIST bymerge)

        self._device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
        logging.info(f'Using {self._device} device for predictions.')

        self._model = EMNISTNet(num_classes=num_classes)
        self._model.load_state_dict(torch.load(model_filename, map_location=self._device))

    def predict(self, inputs):
        """
        inputs: list of 2d numpy array containing N images where pixels lies beetwen 0 and 255
        """
        inputs = [img / 255 for img in inputs] # normalize

        t = torch.tensor(inputs, dtype=torch.float32)
        t.unsqueeze_(0)
        t = t.permute(1,0,2,3)
        logging.debug(f'Tensor for prediction: {t.shape}')

        t.to(self._device)
        preds = self._model(t)
        preds = preds.argmax(dim=1)
        logging.debug(f'Preds shape: {preds.shape}')

        preds_indexes =  list(preds.numpy().astype(int))
        preds_classes = [self._groundtruth[idx] for idx in preds_indexes]
        pred = ''
        return pred.join([str(s) for s in preds_classes])
    