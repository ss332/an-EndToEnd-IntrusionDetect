from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1


# 语言类
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    # 字母小写，去除首尾的空格
    s = unicodeToAscii(s.lower().strip())
    # 正则替换，在标点符号前加空格，单独作为字母处理
    s = re.sub(r"([.!?])", r" \1", s)
    # 一切非ASCII字符用空格代替
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print('读取lines...')

    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print('读取 %s 翻译对' % len(pairs))
    pairs = filterPairs(pairs)
    print('修剪至 %s 翻译对' % len(pairs))
    print('计算单词数...')
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print('计算结果如下：')
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embeding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embeded = self.embeding(input).view(1, 1, -1)
        output = embeded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = MAX_LENGTH

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # [1,1,n_words]
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        # [1,10]
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # [1,1,10]@[1,10,256]=[1,1,256]
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        # [1,256*2]
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # [1,256*2]->[1,256]->[1,1,256]
        output = self.attn_combine(output).unsqueeze(0)
        # MAX(0,X)
        output = F.relu(output)
        # output=[1,1,n_words] hidden=[1,1,256]
        output, hidden = self.gru(output, hidden)
        # [1,n_words]
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    # [len(sentence),1]
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder,decoder,n_iters,print_every=1000,plot_every=100,learning_rate=0.01):
    start=time.time()
    plot_losses=[]
    print_loss_total=0
    plot_loss_total=0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()
    for iter in range(1,n_iters+1):
        training_pair=training_pairs[iter-1]
        input_tensor=training_pair[0]
        target_tensor=training_pair[1]

        loss=train(input_tensor,target_tensor,encoder,decoder,
                   encoder_optimizer,decoder_optimizer,criterion)
        print_loss_total+=loss
        plot_loss_total+=loss
        if iter% print_every==0:
            avg=print_loss_total/print_every
            print_loss_total=0
            print('%s (%d %d%%) %.4f ' % (timeSince(start,iter/n_iters),
                                          iter,iter/n_iters*100,avg))

        if iter % plot_every==0:
            avg=plot_loss_total/plot_every
            plot_losses.append(avg)
            plot_loss_total=0
    showPlot(plot_losses)


def showPlot(points):
    plt.figure()
    fig,ax=plt.subplots()
    loc=ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
evaluateRandomly(encoder1, attn_decoder1)


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("elle a cinq ans de moins que moi .")

evaluateAndShowAttention("elle est trop petit .")

evaluateAndShowAttention("je ne crains pas de mourir .")

evaluateAndShowAttention("c est un jeune directeur plein de talent .")
# 读取lines...
# 读取 135842 翻译对
# 修剪至 10599 翻译对
# 计算单词数...
# 计算结果如下：
# fra 4345
# eng 2803
# ['j attends ma mere .', 'i m waiting for my mother .']
# 4m 32s (- 63m 36s) (5000 6%) 2.8678
# 9m 1s (- 58m 41s) (10000 13%) 2.3004
# 13m 32s (- 54m 9s) (15000 20%) 1.9878
# 18m 32s (- 50m 58s) (20000 26%) 1.7254
# 23m 50s (- 47m 41s) (25000 33%) 1.5089
# 28m 40s (- 43m 0s) (30000 40%) 1.3411
# 33m 51s (- 38m 41s) (35000 46%) 1.2108
# 38m 54s (- 34m 2s) (40000 53%) 1.0855
# 43m 34s (- 29m 2s) (45000 60%) 0.9320
# 48m 14s (- 24m 7s) (50000 66%) 0.8964
# 52m 53s (- 19m 13s) (55000 73%) 0.8243
# 57m 33s (- 14m 23s) (60000 80%) 0.7322
# 62m 16s (- 9m 34s) (65000 86%) 0.6610
# 67m 0s (- 4m 47s) (70000 93%) 0.6083
# 71m 56s (- 0m 0s) (75000 100%) 0.5695
# C:/Users/alice/intrusionDetection/rnn3.py:319: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
#   plt.show()
# > j en ai fini de vous ecouter .
# = i m done listening to you .
# < i m done listening to you . <EOS>
#
# > je suis vraiment decu .
# = i m really disappointed .
# < i m really disappointed . <EOS>
#
# > je vais a l universite .
# = i m going to college .
# < i m going to college . <EOS>
#
# > qu on ne me derange pas !
# = i m not to be disturbed .
# < i m not getting wrong . <EOS>
#
# > je ne suis toujours pas sure .
# = i m still not sure .
# < i m still not sure . <EOS>
#
# > je suis crevee .
# = i m exhausted .
# < i m exhausted . <EOS>
#
# > je suis un homme prudent .
# = i m a careful man .
# < i m a a man . <EOS>
#
# > j ai soif .
# = i am thirsty .
# < i m thirsty . <EOS>
#
# > je suis ton ami .
# = i m your friend .
# < i m your friend . <EOS>
#
# > t es remonte .
# = you re psyched .
# < you re exhausted . <EOS>
#
# input = elle a cinq ans de moins que moi .
# output = she s five years younger than me . <EOS>
# input = elle est trop petit .
# output = she is too hard . <EOS>
# input = je ne crains pas de mourir .
# output = i m not scared to die . <EOS>
# input = c est un jeune directeur plein de talent .
# output = he s a talented young director . <EOS>
#
# Process finished with exit code 0