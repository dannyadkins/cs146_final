from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

def load_dataset(train_file, test_file, batch_size, tokenizer):
    """
    :param fn: filename for the dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for train and test
    :Comment: You don't have to shuffle the test dataset
    """

    train_loader = process_file(train_file, tokenizer, batch_size)
    test_loader = process_file(test_file, tokenizer, batch_size)

    return train_loader, test_loader

def process_file(fn, tokenizer, batch_size=1):

    prefilled_vocab = (not word2id == None)
    f = open(fn, 'rt')
    lines = f.readlines()
    f.close()

    # TODO: split the whole file (including both training and validation
    # data) into words and create the corresponding vocab dictionary.
    vocab = {'*PAD*': 0, '*START*': 1}
    if (prefilled_vocab):
        vocab = word2id
    num_inputs = len(lines)

    inputs = []
    labels = []
    for line in lines:
        label = []
        split_line = line.split()
        input = f'{tokenizer.cls_token}{split_line}{tokenizer.eos_token}'
        label = f'{split_line}{tokenizer.eos_token}'
        input = torch.tensor(input.encode()).unsqueeze(0)
        label = torch.tensor(label.encode()).unsqueeze(0)
        inputs.append(input)
        labels.append(label)

    return DataLoader(list(zip(inputs, labels)),  batch_size=batch_size)
