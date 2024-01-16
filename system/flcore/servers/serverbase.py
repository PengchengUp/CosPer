import torch
import os
import numpy as np
import h5py
import copy
import time
import random


from utils.data_utils import read_client_data
from utils.gradient_utils import flatten_grad, flatten_params

from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.motivation = args.motivation
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 20
        self.div_value = 0.0001
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []
        ########### 列表取值为True和False，True代表对应的客户端被选作攻击者
        self.train_malicious_clients = []
        self.train_random_clients = []
        # self.train_scaling_clients = []

        ########### list of malicious clients
        self.list_malicious_clients = []
        self.list_random_clients = []
        # self.list_scaling_clients = []
        self.benign_clients = []


        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        ####
        self.uploaded_gradients = []
        self.history = []

        self.rs_test_acc = []
        #self.rs_test_auc = []




        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate
        ######## num of malicious clients
        self.malicious_num = args.malicious_num
        self.random_update = args.random_update
        # self.scaling_update = args.scaling_update


        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch

    def set_clients(self, clientObj):
        for i, train_slow, send_slow, train_malicious, train_random  in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients, self.train_malicious_clients, self.train_random_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow,
                            train_malicious=train_malicious,
                            train_random=train_random)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    # random select malicious clients
    def select_malicious_clients(self, malicious_num):
        malicious_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        np.random.seed(0)
        idx_ = np.random.choice(idx, malicious_num)
        for i in idx_:
            malicious_clients[i] = True

        return malicious_clients

    # random select free-rider clients
    def select_random_clients(self, random_update):
        random_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        np.random.seed(0)
        idx_ = np.random.choice(idx, random_update)
        for i in idx_:
            random_clients[i] = True

        return random_clients
    
    # random select train scaling clients
    # def select_scaling_clients(self, scaling_update):
    #     random_clients = [False for i in range(self.num_clients)]
    #     idx = [i for i in range(self.num_clients)]
    #     np.random.seed(0)
    #     idx_ = np.random.choice(idx, scaling_update)
    #     for i in idx_:
    #         random_clients[i] = True

    #     return random_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def set_malicious_clients(self):
        self.train_malicious_clients = self.select_malicious_clients(
            self.malicious_num)

    def set_random_clients(self):
        self.train_random_clients = self.select_random_clients(
            self.random_update)
        
    # def set_scaling_clients(self):
    #     self.train_random_clients = self.select_random_clients(
    #         self.random_update)

    def select_clients(self):
        if self.random_join_ratio:
            num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        ####
        self.uploaded_gradients = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
                client_params = self.get_client_params(client.model)
                self.uploaded_gradients.append(client_params)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples   #聚合权重
        if self.motivation:
            self.history.append(self.uploaded_gradients)              

    ###    
    def get_client_gradients(self, client_model):
        gradients = []
        for param in client_model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().cpu().numpy())
            else:
                gradients.append(None)
        return flatten_grad(gradients)
    
    def get_client_params(self, client_model):
        params = []
        for param in client_model.parameters():
            if param is not None:
                params.append(param.detach().cpu().numpy())
            else:
                params.append(None)
        return flatten_params(params)
    
    def get_client_divergence(self, client_param, global_param):
        '''
        client_param and global_param are flattened vectors in numpy (1-D array)
    
        '''
        divergence = np.linalg.norm(client_param - global_param)
        
        return divergence


    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                #hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    # def test_metrics(self):
    #     if self.eval_new_clients and self.num_new_clients > 0:
    #         self.fine_tuning_new_clients()
    #         return self.test_metrics_new_clients()
        
    #     num_samples = []
    #     tot_correct = []
    #     #tot_auc = []

    #     for c in self.clients:
    #         if c.id in self.benign_clients:#only benign clients
    #             ct, ns = c.test_metrics()
    #             tot_correct.append(ct*1.0)
    #             #tot_auc.append(auc*ns)
    #             num_samples.append(ns)

    #     ids = self.benign_clients
    #     #ids = [c.id for c in self.clients]

    #     return ids, num_samples, tot_correct#, tot_auc

    # def train_metrics(self):
    #     if self.eval_new_clients and self.num_new_clients > 0:
    #         return [0], [1], [0]
        
    #     num_samples = []
    #     losses = []

    #     for c in self.clients:
    #         if c.id in self.benign_clients:#only benign clients
    #             cl, ns = c.train_metrics()
    #             num_samples.append(ns)
    #             losses.append(cl*1.0)

    #     #ids = [c.id for c in self.clients]
    #     ids = self.benign_clients

    #     return ids, num_samples, losses
    
    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        #tot_auc = []
        for c in self.clients:
            ct, ns = c.test_metrics() # test_acc, test_num
            tot_correct.append(ct*1.0)
            #tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = self.benign_clients#[c.id for c in self.benign_clients]

        return ids, num_samples, tot_correct #,tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = self.benign_clients#[c.id for c in self.benign_clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics() #ids, num_samples(list), tot_correct(list), (tot_auc)
        stats_train = self.train_metrics() #ids, num_samples, losses

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        #test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        #aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        #print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Cosine Fairness: {:.4f}".format(self.cosine_fairness(accs)))
        print("Clinets Test Accurancy: {}".format(accs))
        #print("Std Test AUC: {:.4f}".format(np.std(aucs)))


    def cosine_fairness(self, accs):
        # 将列表转换为NumPy数组
        v1 = np.array(accs)
        v2 = np.ones_like(accs)

        # 计算向量的内积
        dot_product = np.dot(v1, v2)

        # 计算向量的模长
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # 计算余弦相似度
        cosine_sim = dot_product / (norm1 * norm2)

        return cosine_sim


    def print_(self, test_acc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        #print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    # def check_done(self, acc_lss, top_cnt=None, div_value=None):
    #     for acc_ls in acc_lss:
    #         if top_cnt != None and div_value != None:
    #             find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
    #             find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
    #             if find_top and find_div:
    #                 pass
    #             else:
    #                 return False
    #         elif top_cnt != None:
    #             find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
    #             if find_top:
    #                 pass
    #             else:
    #                 return False
    #         elif div_value != None:
    #             find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
    #             if find_div:
    #                 pass
    #             else:
    #                 return False
    #         else:
    #             raise NotImplementedError
    #     return True
    def check_done(self, acc_lss, top_cnt=None, div_value=None, target_acc = None):#acc_lss全局模型历史精度
        #target_acc = 0.8 #目标精度 fmnist0.8
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None and target_acc != None:
                top_acc = torch.topk(torch.tensor(acc_ls), 1).values.numpy().tolist()[0]
                print(top_acc)
                if top_acc >= target_acc:
                    find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                    find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                else:
                    find_top = False
                    find_div = False
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None and target_acc != None:
                top_acc = torch.topk(torch.tensor(acc_ls), 1).values.numpy().tolist()[0]
                print(top_acc)
                print(target_acc)

                if top_acc >= target_acc:
                    find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                else:
                    find_top = False
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False,
                            train_malicious=False,
                            train_random=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            for e in range(self.fine_tuning_epoch):
                client.train()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        #tot_auc = []
        for c in self.new_clients:
            ct, ns = c.test_metrics()
            tot_correct.append(ct*1.0)
            #tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct#, tot_auc
