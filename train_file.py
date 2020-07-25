import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
# from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import GPUtil

"""
To install spacy languages do:
python -m spacy download en
python -m spacy download de
"""
# spacy_ger = spacy.load("de")
spacy_eng = spacy.load("en")


# def tokenize_ger(text):
#     return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

# train_data, valid_data, test_data = Multi30k.splits(
#     exts=(".de", ".en"), fields=(german, english)
# )

print("Start: tokenizing the train and validation set")
data_fields = [('source', english), ('target', english)]

train, val = TabularDataset.splits(path='./', train='train.csv', validation='val.csv', format='csv', fields=data_fields)
print("train ",len(train))
print("val ",len(val))
print("Done with creation of DataSet Object")

english.build_vocab(train, max_size=10000, min_freq=2)

print(len(english.vocab))

DROPOUT_TOKENS = ["a", "an", "the", "'ll", "'s", "'m", "'ve", "to"]  # Add "to"

REPLACEMENTS = ["their", "there", "then", "than"]


CORRECTIVE_TOKENS = DROPOUT_TOKENS + REPLACEMENTS


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # x: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        encoder_states, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))
        del embedding
        torch.cuda.empty_cache()
        return encoder_states, hidden, cell
    
class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        # x: (1, N) where N is the batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)

        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, hidden_size)
        
        del x, embedding,energy, attention,outputs
        torch.cuda.empty_cache()
#         for i in range(100000000):
#             continue
        return predictions, hidden, cell
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        encoder_states, hidden, cell = self.encoder(source)

        # First input will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # At every time step use encoder_states and update hidden, cell
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            # Store prediction for current time step
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        del x, encoder_states, hidden, cell,target_vocab_size
        torch.cuda.empty_cache()
        return outputs
    
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1" if False else "cpu")
device2 = torch.device("cuda:2" if False else "cpu")
load_model = False
save_model = True

# Training hyperparameters
num_epochs = 100
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
input_size_encoder = len(english.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 1
enc_dropout = 0.0
dec_dropout = 0.0

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step = 0
# GPUtil.showUtilization()

train_iterator = BucketIterator(
    train,
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.source),
    device=device2
)


encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)
GPUtil.showUtilization()
decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device2)

model = Seq2Seq(encoder_net, decoder_net).to(device)
# model = torch.nn.DataParallel(model, device_ids=[1,2]).to(device)
# GPUtil.showUtilization()
# model = model.module()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# model = model.module
pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = (
    "I don't want there food"
    "the Cardinals did better then the Cubs in the offseason"
)

for epoch in range(num_epochs):
    print("[Epoch {} / {}]".format(epoch,num_epochs))

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(
        model, sentence, english, english, device, max_length=50
    )

    print("Translated example sentence: \n {}".format(translated_sentence))

    model.train()
#     GPUtil.showUtilization()

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.source.to(device)
        target = batch.target.to(device)

        # Forward prop
        output = model(inp_data, target)
#         GPUtil.showUtilization()
        
        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
#         GPUtil.showUtilization()
        loss = criterion(output, target)
        
        del output, inp_data, target
        torch.cuda.empty_cache()
#         for i in range(100000000):
#             continue
        # Back prop
        loss.mean().backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1
        del loss
        torch.cuda.empty_cache()
#         for i in range(100000000):
#             continue
# running on entire test data takes a while
score = bleu(val[1:100], model, english, english, device)
print("Bleu score {}".format(score * 100))