from flcore.clients.clientcosper import clientCosPer
from flcore.servers.serverbase import Server
from threading import Thread


class CosPer(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_malicious_clients()
        self.set_random_clients()
        self.set_clients(clientCosPer)

        for id, value in enumerate(self.train_malicious_clients):
            if value:
                self.list_malicious_clients.append(id)
        for id, value in enumerate(self.train_random_clients):
            if value:
                self.list_random_clients.append(id)
        self.benign_clients = list(set(list(range(self.num_clients))).difference(self.list_malicious_clients + self.list_random_clients))

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"\nMalicious clients: {self.list_malicious_clients}")
        print(f"\nFree riders: {self.list_random_clients}")
        print(f"\nBenign clients: {self.benign_clients}")
        print("Finished creating server and clients.")

        # self.load_model()


    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
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

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(self.rs_test_acc.index(max(self.rs_test_acc)), end=':')
        print(max(self.rs_test_acc))

        self.save_results()

        self.eval_new_clients = True
        self.set_new_clients(clientCosPer)
        print(f"\n-------------Fine tuning round-------------")
        print("\nEvaluate new clients")
        self.evaluate()
