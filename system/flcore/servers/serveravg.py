import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import pandas as pd
import numpy as np
from utils.gradient_utils import cossimilarity, flatten_params, mutual_information, aggregate_parameters, distance
from scipy.stats import ttest_ind, ttest_1samp



class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()

        # select malicious clients
        self.set_malicious_clients()
        self.set_random_clients()
        self.set_clients(clientAVG)

        for id, value in enumerate(self.train_malicious_clients):
            if value:
                self.list_malicious_clients.append(id)
        for id, value in enumerate(self.train_random_clients):
            if value:
                self.list_random_clients.append(id)
        self.benign_clients = list(
            set(list(range(self.num_clients))).difference(self.list_malicious_clients + self.list_random_clients))

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"\nMalicious clients: {self.list_malicious_clients}")
        print(f"\nFree riders: {self.list_random_clients}")
        print(f"\nBenign clients: {self.benign_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


        

    def train(self):
        ###
        global_history = []
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            ###
            global_history.append(flatten_params(self.get_client_params(self.global_model)))

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt, target_acc = 0.8):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(self.rs_test_acc.index(max(self.rs_test_acc)), end=':')
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        self.eval_new_clients = True
        self.set_new_clients(clientAVG)
        print(f"\n-------------Fine tuning round-------------")
        print("\nEvaluate new clients")
        self.evaluate()
        ####
        '''
        if self.motivation:
            cos = []
            distance = []
            cos_client = []
            distance_client = []
            for c in range(self.num_clients):
                for t in range(1,self.global_rounds+1):
                    cos_client.append(cossimilarity(self.history[t][c], self.history[t-1][c]))
                    distance_client.append(np.linalg.norm(self.history[t][c] - self.history[t-1][c]))
                cos.append(cos_client)
                distance.append(distance_client)
                cos_client = []
                distance_client = []
            print('cos:',cos)
            print('norm:',distance)
        '''
        ###
        if self.motivation:
            print(len(self.history))
            benign_model_history, malicious_model_history = aggregate_parameters(self.history)
            print(len(benign_model_history),len(malicious_model_history))

            #t_statistic_d, p_value_d = ttest_ind(benign_model_history, malicious_model_history)
            # print('Double Tes:')
            # print("t_statistic:", t_statistic_d)
            # print("p_value:", p_value_d)
            
            distance_history = []
            for wb, wm in zip(benign_model_history,malicious_model_history):
                d = distance(np.array(wb), np.array(wm))
                distance_history.append(d)
            t_statistic_s, p_value_s = ttest_1samp(np.array(distance_history), popmean=0)
            print("distance_history:", distance_history)
            print('Single Tes:')
            print("t_statistic:", t_statistic_s)
            print("p_value:", p_value_s)

    def aggregate_parameters(self):
        benign_centers= []
        malicious_centers = []
        for t in range(len(self.history)):
            benign_center = self.history[t][1]
            malicious_center = self.history[t][0]
            for i in self.benign_clients:
                benign_center = benign_center / 10 + self.history[t][i] / 10

            benign_centers.append(benign_center)

            for j in self.list_malicious_clients + self.list_random_clients:
                malicious_center = malicious_center / 10 + self.history[t][j] /10

            malicious_centers.append(malicious_center)

        return benign_centers, malicious_centers
        
 




