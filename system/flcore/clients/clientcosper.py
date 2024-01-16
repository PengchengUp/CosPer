import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
import random


from sklearn.preprocessing import label_binarize
from sklearn import metrics

class clientCosPer(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.model_per = copy.deepcopy(self.model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.7)
        self.optimizer_per = torch.optim.SGD(self.model_per.parameters(), lr=self.learning_rate, momentum=0.7)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        #self.optimizer_per = torch.optim.SGD(self.model_per.parameters(), lr=self.learning_rate)
        #指数衰减调整学习率
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        # malicious lients load label poisoned data
        if self.train_malicious:
            trainloader = self.load_malicious_train_data()
        else:
            trainloader = self.load_train_data()
        
        start_time = time.time()

        #self.model.to(self.device)
        self.model.train()
        #self.model_per.to(self.device)
        self.model_per.train()

        max_local_steps = self.local_epochs
        beta = (1 - 1 / max_local_steps)

        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)
            #生成一个随机整数，其值在1和(max_local_steps // 2)之间（左闭右开区间）
        simularity = 0
        if self.train_random:
            self.random_update(self.model)
            similarity = random.random()
        else:
            for step in range(max_local_steps):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))

                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # #获取global model的训练梯度
                    grads = [param.grad for param in self.model.parameters()]
                    self.optimizer.step()

                    output_per = self.model_per(x)
                    loss_per = self.loss(output_per, y)
                    self.optimizer_per.zero_grad()
                    loss_per.backward()
                    # # 获取personalized model的训练梯度
                    grads_per = [param.grad for param in self.model_per.parameters()]
                    self.optimizer_per.step()
                # 获取global model的训练梯度
                #grads = [param.grad for param in self.model.parameters()]
                # 获取global model的训练梯度
                #grads_per = [param.grad for param in self.model_per.parameters()]
                similarity_s = self.similarity_update(grads, grads_per)
                similarity = (1 - beta) * simularity + beta * similarity_s
        for lp, p in zip(self.model_per.parameters(), self.model.parameters()):
            lp.data = similarity * p + (1 - similarity) * lp

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def process_grad(self,grads):
        '''
            Args:
                grads: grad
            Return:
                a flattened grad in numpy (1-D array)
            '''
        flattened_grads = grads[0].cpu().numpy()
        for i in range(1, len(grads)):
            flattened_grads = np.append(flattened_grads, grads[i].cpu().numpy())  # output a flattened array

        return flattened_grads

    def similarity_update(self, grads, grads_per):
        '''Returns the cosine similarity between grads and grads_per
            '''
        a = self.process_grad(grads)
        b = self.process_grad(grads_per)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        return dot_product * 1.0 / (norm_a * norm_b)


    def test_metrics(self):

        testloaderfull = self.load_test_data()
        self.model_per.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model_per(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        #auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num#, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model_per.train()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output_per = self.model_per(x)
                loss_per = self.loss(output_per, y)
                train_num += y.shape[0]
                losses += loss_per.item() * y.shape[0]

        return losses, train_num
