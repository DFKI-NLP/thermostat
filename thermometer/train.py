import torch
from ignite.handlers import ModelCheckpoint


def load_checkpoint(path_model, model):
    checkpoint = torch.load(path_model)
    to_load = {'model': model}
    ModelCheckpoint.load_objects(to_load=to_load,
                                 checkpoint=checkpoint)  # overwrite pretrained weights w/ fine-tuned weights
