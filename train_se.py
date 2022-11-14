# -*- coding: utf-8 -*-
import os
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
from bucket_iterator_se import BucketIterator
from sklearn import metrics
from data_utils import ABSADatesetReader
from models import LSTM, MLGCN_SE


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, opt=opt)

        self.opt.max_len = max([len(t['text_indices']) for t in absa_dataset.train_data.data]+[len(t['text_indices']) for t in absa_dataset.test_data.data])

        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True, sort=True, max_len=self.opt.max_len)

        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False, max_len=self.opt.max_len)

        self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        self._print_args()
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:{}MB'.format(torch.cuda.memory_allocated(device=opt.device.index)/1024/1024))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer):
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0
        train_f1 = []
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            train_loss_epoch = 0
            n_correct, n_total = 0, 0
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                inputs.append(sample_batched['word_height'])
                targets = sample_batched['polarity'].to(self.opt.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, test_f1 = self._evaluate_acc_f1()
                    train_f1.append(float(test_f1))
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            torch.save(self.model.state_dict(), './state_dict/'+self.opt.dataset+'_lr-'+str(self.opt.learning_rate)+'_tree-'+str(self.opt.tree)+'.pkl')
                            print('>>> best model saved.')
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc, test_acc, test_f1))
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 10:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0
            # train_loss.append(float(train_loss_epoch/self.train_data_loader.batch_len))
        # drawPlot([train_f1],'{}_loss_{}_{}.png'.format(self.opt.dataset, self.opt.num_epoch, self.opt.learning_rate), 'f1', ['train'])
        return max_test_acc, max_test_f1

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                t_inputs.append(t_sample_batched['word_height'])
                t_targets = t_sample_batched['polarity'].to(opt.device)
                t_outputs = self.model(t_inputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1

    def run(self, repeats=3):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        
        if not os.path.exists('log/'):
            os.mkdir('log/')

        max_test_acc_avg = 0
        max_test_f1_avg = 0
        max_acc = 0
        max_f1 = 0
        for i in range(repeats):
            print('repeat: ', (i+1))
            self._reset_params()
            _params = filter(lambda p: p.requires_grad, self.model.parameters())    # filter(func, seq): 筛选出seq中使得func成立的元素
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            max_test_acc, max_test_f1 = self._train(criterion, optimizer)
            print('max_test_acc: {0}     max_test_f1: {1}'.format(max_test_acc, max_test_f1))
            max_acc = max_test_acc if max_test_acc > max_acc else max_acc
            max_f1 = max_test_f1 if max_test_f1 > max_f1 else max_f1
            max_test_acc_avg += max_test_acc
            max_test_f1_avg += max_test_f1
            print('#' * 100)
        print('repeats:', repeats)
        print("max_test_acc_avg:", max_test_acc_avg / repeats)
        print("max_test_f1_avg:", max_test_f1_avg / repeats)
        print('max_acc:', max_acc)
        print('max_f1:', max_f1)


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='mlgcn', type=str)
    parser.add_argument('--dataset', default='rest14', type=str, help='twitter, rest14, lap14, rest15, rest16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--l2reg', default=0.0001, type=float)     # L2正则化系数，给优化器optimizer使用的，但是不知道干啥
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--pos_dim', default=0, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=30, type=int)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--rnn_drop', default=0.3, type=float, help='RNN dropout rate.')
    parser.add_argument('--gcn_drop', default=0.2, type=float, help='GCN dropout rate.')
    parser.add_argument('--attention_heads', default=1, type=int)
    parser.add_argument('--tree', default=False, type=bool)
    parser.add_argument('--repeats', default=10, type=int)
    parser.add_argument('--max_len', default=85, type=int)
    opt = parser.parse_args()

    model_classes = {
        'lstm': LSTM,
        'mlgcn': MLGCN_SE,
    }
    input_colses = {
        'lstm': ['text_indices'],
        'mlgcn': ['text_indices', 'aspect_indices', 'left_indices', 'pos_indices', 'dependency_graph', 'asp_post'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run(repeats=opt.repeats)
