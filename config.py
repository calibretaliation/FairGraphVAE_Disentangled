class Config():
    def __init__(self):

        # Model config
        self.model_debug = True
        self.num_nodes = 5
        self.num_feats = 3
        self.latent_dim_S = 128
        self.latent_dim_Y = 128
        self.gcn_hidden_dim = 512
        self.num_labels = 2
        self.device = 'cuda:2'
        self.pool = "sum"

        #Data config
        self.batch_size = 10

        #Training config
        self.learning_rate = 1e-5
        self.LR_milestones = [500, 1000]
        self.train_epoch = 500
        self.log_epoch = 10