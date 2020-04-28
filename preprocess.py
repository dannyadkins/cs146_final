from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

def load_dataset(train_file, test_file, batch_size, tokenizer, seq_len):
    """
    :param fn: filename for the dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for train and test
    :Comment: You don't have to shuffle the test dataset
    """

    train_loader = process_file(train_file, tokenizer, batch_size)
    test_loader = process_file(test_file, tokenizer, batch_size)

    return train_loader, test_loader

def process_file(fn, tokenizer, batch_size=1, seq_len=50):
    f = open(fn, 'rt')
    lines = f.readlines()
    f.close()

    # TODO: split the whole file (including both training and validation
    # data) into words and create the corresponding vocab dictionary.

    inputs = []
    labels = []
    for line in lines:
        label = []
        input = f'{tokenizer.cls_token}{line}{tokenizer.eos_token}'
        label = f'{line}{tokenizer.eos_token}'
        input = torch.tensor(tokenizer.encode(input))
        label = torch.tensor(tokenizer.encode(label))
        inputs.append(input)
        labels.append(label)

    tensor_inputs = []
    for input in inputs:
        tensor_inputs.append(torch.Tensor(input.float()))

    tensor_labels = []
    for label in labels:
        tensor_labels.append(torch.Tensor(label.float()))



    inputs = torch.tensor(pad_sequence(tensor_inputs, batch_first=True))[:, :seq_len]
    labels = torch.tensor(pad_sequence(tensor_labels, batch_first=True))[:, :seq_len]


    return DataLoader(list(zip(inputs, labels)),  batch_size=batch_size)
