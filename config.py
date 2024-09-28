class Config():
    def __init__(self):

        # Model config
        self.num_nodes = 5
        self.num_feats = 3
        self.num_sensitive_class = 2
        self.num_labels = 2
        self.latent_dim_S = 128
        self.latent_dim_Y = 128
        self.gcn_hidden_dim = 512
        self.device = 'cuda:0'
        self.pool = "attention"
        self.dataset_name = "nba"
        #Data config
        self.batch_size = 10
        self.train_size = 0.75
        self.data_path = "data"
        #Training config
        self.learning_rate = 1e-4
        self.LR_milestones = [500, 1000]
        self.train_epoch = 10001
        self.log_epoch = 100
        self.lambda_hgr = 1e4
        self.efl_gamma =1e4