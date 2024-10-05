class Config():
    def __init__(self):
        # Model config
        self.num_nodes = 5
        self.num_feats = 3
        self.num_sensitive_class = 2
        self.num_labels = 2
        self.latent_dim_S = 64
        self.latent_dim_Y = 64
        self.gcn_hidden_dim = 512
        self.device = 'cuda:0'
        self.pool = "attention"
        self.dataset_name = "nba"
        #Data config
        self.batch_size = 500
        self.train_size = 0.75
        self.data_path = "data"
        self.random_walk_length = 50
        self.num_random_walk_sample = 20
        #Training config
        self.learning_rate = 1e-4
        self.LR_milestones = [500, 1000]
        self.train_epoch = 10001
        self.log_epoch = 10
        self.lambda_hgr = 1e4
        self.efl_gamma =1e4
        self.grid = None
    def show(self, logger = None):
        if logger is not None:
            logger.info(f"TRAINING CONFIG:\nnum_nodes:\t{self.num_nodes}\nnum_feats:\t{self.num_feats}\nnum_sensitive_class:\t{self.num_sensitive_class}\nnum_labels:\t{self.num_labels}\nlatent_dim_S:\t{self.latent_dim_S}\nlatent_dim_Y:\t{self.latent_dim_Y}\ngcn_hidden_dim:\t{self.gcn_hidden_dim}\ndevice:\t{self.device}\npool:\t{self.pool}\ndataset_name:\t{self.dataset_name}\nbatch_size:\t{self.batch_size}\ntrain_size:\t{self.train_size}\ndata_path:\t{self.data_path}\nrandom_walk_length:\t{self.random_walk_length}\nnum_random_walk_sample:\t{self.num_random_walk_sample}\nlearning_rate:\t{self.learning_rate}\nLR_milestones:\t{self.LR_milestones}\ntrain_epoch:\t{self.train_epoch}\nlog_epoch:\t{self.log_epoch}\nlambda_hgr:\t{self.lambda_hgr}\nefl_gamma:\t{self.efl_gamma}")
        else:
            print(f"TRAINING CONFIG:\nnum_nodes:\t{self.num_nodes}\nnum_feats:\t{self.num_feats}\nnum_sensitive_class:\t{self.num_sensitive_class}\nnum_labels:\t{self.num_labels}\nlatent_dim_S:\t{self.latent_dim_S}\nlatent_dim_Y:\t{self.latent_dim_Y}\ngcn_hidden_dim:\t{self.gcn_hidden_dim}\ndevice:\t{self.device}\npool:\t{self.pool}\ndataset_name:\t{self.dataset_name}\nbatch_size:\t{self.batch_size}\ntrain_size:\t{self.train_size}\ndata_path:\t{self.data_path}\nrandom_walk_length:\t{self.random_walk_length}\nnum_random_walk_sample:\t{self.num_random_walk_sample}\nlearning_rate:\t{self.learning_rate}\nLR_milestones:\t{self.LR_milestones}\ntrain_epoch:\t{self.train_epoch}\nlog_epoch:\t{self.log_epoch}\nlambda_hgr:\t{self.lambda_hgr}\nefl_gamma:\t{self.efl_gamma}")
global config
config = Config()