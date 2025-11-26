import os
import csv
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Default device - will be set by set_device()
device = torch.device("cpu")

def set_device(device_name):
    """Set the global device for tensor operations."""
    global device
    device = torch.device(device_name)
    print(f"Using device: {device}")

def var(tensor):
    """Move tensor to the configured device."""
    return tensor.to(device)


def get_tensor(z):
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))


class Logger:
    def __init__(self, log_path):
        self.writer = SummaryWriter(log_path)

    def print(self, name, value, episode=-1, step=-1):
        string = "{} is {}".format(name, value)
        if episode > 0:
            print("Episode:{}, {}".format(episode, string))
        if step > 0:
            print("Step:{}, {}".format(step, string))

    def write(self, name, value, index):
        self.writer.add_scalar(name, value, index)


def _is_update(episode, freq, ignore=0, rem=0):
    if episode != ignore and episode % freq == rem:
        return True
    return False


def record_experience_to_csv(args, experiment_name, csv_name="experiments.csv"):
    # append DATE_TIME to dict
    d = vars(args)
    d["date"] = experiment_name

    if os.path.exists(csv_name):
        # Save Dictionary to a csv
        with open(csv_name, "a") as f:
            w = csv.DictWriter(f, list(d.keys()))
            w.writerow(d)
    else:
        # Save Dictionary to a csv
        with open(csv_name, "w") as f:
            w = csv.DictWriter(f, list(d.keys()))
            w.writeheader()
            w.writerow(d)


def listdirs(directory):
    return [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]
