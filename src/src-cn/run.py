import numpy as np
import sys
import argparse
import pickle
import math
from copy import deepcopy
# sys.path.append('../')

import torch.nn as nn
from torch.optim import Adam,SGD,Adadelta,Adagrad
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torchnet import meter
# from fastNLP import Trainer
# from fastNLP.core.losses import CrossEntropyLoss
# from fastNLP.core.metrics import AccuracyMetric
# from fastNLP.core import optimizer as fastnlp_optim
# from fastNLP.core.callback import EarlyStopCallback
# from fastNLP import Batch
from config import Config
from dataset import JokeData
from model import JokeModel

parser = argparse.ArgumentParser()

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--generate', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gentest', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='Adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout', type=float, default=0,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=128,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--embed_size', type=int, default=128,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=256,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_seq_len', type=int, default=50,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_gen_len', type=int, default=50,
                                help='max length of passage')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()

def train(conf,args=None):
    if conf.vocab_path is not None:
        with open(conf.vocab_path, 'rb') as f:
            jdata = pickle.load(f)
    else:
        jdata = JokeData(conf)
        jdata.get_vocab()
    print("Data Ready")
    print("voab_size: {}".format(jdata.vocab_size))
    if conf.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = JokeModel(jdata.vocab_size,conf,device)

    train_data = jdata.train_data
    test_data = jdata.test_data

    train_data = torch.from_numpy(np.array(train_data['pad_words']))
    dev_data = torch.from_numpy(np.array(test_data['pad_words']))

    dataloader = DataLoader(train_data, batch_size=conf.batch_size,shuffle=True,num_workers=conf.num_workers)
    devloader = DataLoader(dev_data,batch_size=conf.batch_size,shuffle=True,num_workers=conf.num_workers)
    
    optimizer = Adam(model.parameters(),lr = conf.learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()


    if conf.load_best_model:
        model.load_state_dict(torch.load(conf.beat_model_path))
    if conf.use_gpu:
        model.cuda()
        criterion.cuda()
    step=0
    bestppl = 1e9
    early_stop_controller=0
    for epoch in range(conf.n_epochs):
        losses=[]
        loss_meter.reset()
        model.train()
        for i,data in enumerate(dataloader):
            data = data.long().transpose(1,0).contiguous()
            if conf.use_gpu:
                data = data.cuda()
            input,target = data[:-1,:],data[1:,:]
            optimizer.zero_grad()
            output, _= model(input)
            loss = criterion(output, target.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            loss_meter.add(loss.item())
            step+=1
            if step%100==0:
                print("epoch_%d_step_%d_loss:%0.4f" % (epoch+1, step,loss.item()))
        train_loss = float(loss_meter.value()[0])
        
        model.eval()
        for i,data in enumerate(devloader):
            data = data.long().transpose(1,0).contiguous()
            if conf.use_gpu:
                data = data.cuda()
            input,target = data[:-1,:],data[1:,:]
            output, _= model(input)
            loss = criterion(output, target.view(-1))
            loss_meter.add(loss.item())
        ppl = math.exp(loss_meter.value()[0])
        print("epoch_%d_loss:%0.4f , ppl:%0.4f" % (epoch+1,train_loss,ppl) )

        if epoch % conf.save_every == 0:
            torch.save(model.state_dict(),"{0}_{1}_{2}".format(conf.model_prefix,epoch,str(ppl)))
            with open ("{0}out_{1}".format(conf.out_path,epoch),'w',encoding='utf-8') as fout:
                gen_joke = generate(model, jdata.vocab, conf)
                fout.write("".join(gen_joke)+'\n\n')
        if ppl<bestppl:
            bestppl = ppl
            early_stop_controller = 0
            
            torch.save(model.state_dict(),"{0}".format(conf.best_model_path))
        else:
            early_stop_controller += 1
        if early_stop_controller>conf.patience:
            print("early stop.")
            break

def generate(model, vocab, conf):
    start_words = conf.start_words
    if not isinstance(start_words, list):
        results = list(start_words)
    else:
        results = start_words
    start_len = len(start_words)
    input = Variable(torch.Tensor([vocab.to_index('<START>')]).view(1,1).long())
    if conf.use_gpu:input=input.cuda()
    hidden = None
    if conf.prefix_words:
        for word in conf.prefix_words:
            output,hidden = model(input,hidden)
            input = Variable(input.data.new([vocab.to_index(word)])).view(1,1)
    
    for i in range(conf.max_gen_len):
        output,hidden = model(input,hidden)
        if i < start_len:
            next_word = results[i]
            input = Variable(input.data.new([vocab.to_index(next_word)])).view(1,1)
        else:
            next_word_id = output.argmax(dim = 1)
            # prob = torch.exp(output[0]/conf.tao)
            # next_word_id = prob.multinomial(1)
            
            # next_word = vocab.to_word(next_word_id.cpu().numpy().tolist()[0])
            next_word = vocab.to_word(next_word_id.item())
            results.append(next_word)
            input = Variable(input.data.new([vocab.to_index(next_word)])).view(1,1)
        if next_word == '<EOS>':
            del results[-1]
            break
    print(" ".join(results))
    return results

def generate_p(model, vocab, conf):
    start_words = conf.start_words
    if not isinstance(start_words, list):
        results = list(start_words)
    else:
        results = start_words
    start_len = len(start_words)
    input = Variable(torch.Tensor([vocab.to_index('<START>')]).view(1,1).long())
    if conf.use_gpu:input=input.cuda()
    hidden = None
    if conf.prefix_words:
        for word in conf.prefix_words:
            output,hidden = model(input,hidden)
            input = Variable(input.data.new([vocab.to_index(word)])).view(1,1)
    
    for i in range(conf.max_gen_len):
        output,hidden = model(input,hidden)
        if i < start_len:
            next_word = results[i]
            input = Variable(input.data.new([vocab.to_index(next_word)])).view(1,1)
        else:
            next_word_id = output.argmax(dim = 1)
            prob = torch.exp(output[0]/conf.tao)
            next_word_id = prob.multinomial(1)
            
            next_word = vocab.to_word(next_word_id.cpu().numpy().tolist()[0])
            # next_word = vocab.to_word(next_word_id.item())
            results.append(next_word)
            input = Variable(input.data.new([vocab.to_index(next_word)])).view(1,1)
        if next_word == '<EOS>':
            del results[-1]
            break
    print(" ".join(results))
    return results

def run():
    conf = Config()
    args = parse_args()
    #args.train = True
    if conf.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.train:
        train(conf,args)
    if args.generate:
        if conf.best_model_path is not None and conf.vocab_path is not None:
            with open(conf.vocab_path, 'rb') as f:
                jdata = pickle.load(f)
            model = JokeModel(jdata.vocab_size,conf,device)
            model.load_state_dict(torch.load(conf.best_model_path))
            if conf.use_gpu:
                model.cuda()
            result = generate_p(model, jdata.vocab, conf)
            print("生成完毕！")
    if args.gentest:
        if conf.best_model_path is not None and conf.vocab_path is not None:
            with open(conf.vocab_path, 'rb') as f:
                jdata = pickle.load(f)
            model = JokeModel(jdata.vocab_size,conf,device)
            model.load_state_dict(torch.load(conf.best_model_path))
            if conf.use_gpu:
                model.cuda()
            x = 0
            while (x < 100):
                print("输入开始句：")
                conf.start_words = input()
                result = generate_p(model, jdata.vocab, conf)
                print("生成完毕！")
                x += 1

if __name__ == "__main__":
    run()
