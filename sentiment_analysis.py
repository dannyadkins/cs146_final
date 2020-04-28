from comet_ml import Experiment

import torch
from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer
longformer_config = LongformerConfig.from_pretrained('longformer-base-4096/')

from preprocess import *
import argparse
from model import Transformer
from torch import nn
from tqdm import tqdm

hyperparams = {
    "batch_size": 100,
    "num_epochs": 1,
    "learning_rate": 0.001,
    "model_dim": 512,
    "embedding_size": 256,
    "num_heads": 8,
    "num_sublayers": 6,
}

experiment = Experiment(project_name="longformer")
experiment.log_parameters(hyperparams)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Train the Model
def train(model, train_loader, experiment, hyperparams):
    print("Training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    loss = nn.CrossEntropyLoss(ignore_index=0)

    with experiment.train():
        for epoch in range(0, hyperparams["num_epochs"]):
            for (inputs, labels) in tqdm(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                attention_mask = gen_attention_mask(inputs)
                input_ids, attention_mask = pad_to_window_size(inputs, attention_mask, longformer_config.attention_window[0], tokenizer.pad_token_id)

                predictions = model(input_ids, attention_mask=attention_mask)[0]

                labels = torch.flatten(labels)

                l = loss(predictions, labels.long())
                print(" Loss:", l)

                optimizer.zero_grad()
                l.backward()
                optimizer.step()

        # Log perplexity to Comet.ml using experiment.log_metric


# Test the Model
def test(model, test_loader, experiment, hyperparams):
    total_loss = 0
    word_count = 0
    correct = 0

    loss = nn.CrossEntropyLoss(ignore_index=0)

    with experiment.test():
        for (inputs, labels) in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = model.forward(inputs.long())
            orig_shape = predictions.size()

            predictions = predictions.view(-1, model.vocab_size)

            l = loss(predictions, torch.flatten(labels).long())

            print(" Loss:", l)

            total_loss += (l * input_lengths).sum().detach().numpy()
            word_count += input_lengths.sum().detach().numpy()

            decoded = torch.argmax(predictions.view(orig_shape), dim=2)
            num_correct = (decoded == labels).sum().float()
            print("Correct: ", num_correct)
            correct += num_correct.detach().numpy()

        per_word_loss = (total_loss)/word_count
        perplexity = torch.exp(torch.tensor(per_word_loss)).detach().numpy()


        accuracy = (torch.tensor(correct)/word_count).detach().numpy()

        print("perplexity:", perplexity)
        print("accuracy:", accuracy)
        experiment.log_metric("perplexity", perplexity)
        experiment.log_metric("accuracy", accuracy)

def gen_attention_mask(inputs):
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
    attention_mask[:, [1, 4, 21,]] =  2
    return attention_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()
    # Data Loader (Input Pipeline)

    # Initialize your transformer using the hyper-parameters

    longformer_config.attention_mode = 'sliding_chunks'
    longformer_model = Longformer.from_pretrained('longformer-base-4096/', config=longformer_config)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.max_len = longformer_model.config.max_position_embeddings
    transformer_model = Transformer(
        model_dim=hyperparams["model_dim"],
        embedding_size=hyperparams["embedding_size"],
        num_heads=hyperparams["num_heads"], vocab_size=tokenizer.vocab_size, num_sublayers=hyperparams["num_sublayers"],  seq_len=longformer_config.attention_window[0]).to(device)

    train_loader, test_loader = load_dataset(args.train_file, args.test_file, batch_size=hyperparams["batch_size"], tokenizer=tokenizer, seq_len=longformer_config.attention_window[0])

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load("./model.pt"))
    if args.train:
        print("running training loop...")
        train(longformer_model, train_loader, experiment, hyperparams)
    if args.test:
        print("running testing loop...")
        test(model, test_loader, experiment, hyperparams)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), "./model.pt")
