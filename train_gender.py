import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from utils import load_data
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from mydataset import GermanDataset, CreditDataset, BailDataset,PokeczDataset,NbaDataset
import argparse
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
import time
import logging
from model import GCNEncoder_s,GraphDecoder_s
from hgr import hgr_correlation
import numpy as np

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(hidden_dim, latent_dim, cached=True)
        self.conv_logstd = GCNConv(hidden_dim, latent_dim, cached=True)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class GenderClassifier(nn.Module):
    def __init__(self, latent_dim, gender_dim):
        super(GenderClassifier, self).__init__()
        self.liner1 = nn.Linear(latent_dim, latent_dim // 2)
        self.liner2 = nn.Linear(latent_dim // 2, latent_dim // 4)
        self.fc_gender = nn.Linear(latent_dim // 4, gender_dim)

    def forward(self, x):
        x = self.liner1(x).relu()
        x = self.liner2(x).relu()
        x = self.fc_gender(x)
        return x
    
class LabelClassifier(nn.Module):
    def __init__(self, latent_dim):
        super(LabelClassifier, self).__init__()
        self.fc_gender = nn.Linear(latent_dim, 1)

    def forward(self, z):
        logits = self.fc_gender(z)
        return logits


class InnerProductDecoder(torch.nn.Module):
    def forward(self,z,edge_index,sigmoid: bool = True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            edge_index (torch.Tensor): The edge indices.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(GraphDecoder, self).__init__()
        self.deconv = GCNConv(latent_dim, output_dim) # x

    def forward(self, z, edge_index):
        return self.deconv(z, edge_index)

class GumbelSoftmaxVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, gender_dim):
        super(GumbelSoftmaxVAE, self).__init__()
        self.encoder = VariationalGCNEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder_x = GraphDecoder(latent_dim, output_dim)
        self.decoder_edge = InnerProductDecoder()
        self.gender_classifier = GraphDecoder(latent_dim, gender_dim)
        self.label_classifier = LabelClassifier(latent_dim)

    def recon_edge_loss(self, z, pos_edge_index, neg_edge_index = None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder_edge(z, pos_edge_index)[1] + 1e-15).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder_edge(z, neg_edge_index)[1] + 1e-15).mean()

        return pos_loss + neg_loss
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def kl_loss(self, mu, logstd):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
                set to :obj:`None`, uses the last computation of :math:`\mu`.
                (default: :obj:`None`)
            logstd (torch.Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`. (default: :obj:`None`)
        """
        logstd =  logstd.clamp(max=10)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def gumbel_softmax(self, logits, temperature=1.0, hard=False):
        u = torch.rand_like(logits)
        g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        y = logits + g
        y_soft = F.softmax(y / temperature, dim=-1)
        if hard:
            _, k = y_soft.max(-1)
            y_hard = torch.zeros_like(logits).scatter_(-1, k.view(-1, 1), 1.0)
            y = (y_hard - y_soft).detach() + y_soft
        else:
            y = y_soft
        return y

    def forward(self, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder_x(z, edge_index)
        logits_gender = self.gender_classifier(z,edge_index)
        logits_label = self.label_classifier(z)
        gender = self.gumbel_softmax(logits_gender, temperature=1.0, hard=True)
        return recon_x, gender, logits_label, mu, logvar, z


class GAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(GAE, self).__init__()
        self.encoder = GCNEncoder_s(input_dim, hidden_dim, latent_dim)
        self.decoder_x = GraphDecoder_s(latent_dim, output_dim)
        self.decoder_edge = InnerProductDecoder()

    def recon_edge_loss(self, z, pos_edge_index, neg_edge_index = None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder_edge(z, pos_edge_index)[1] + 1e-15).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder_edge(z, neg_edge_index)[1] + 1e-15).mean()

        return pos_loss + neg_loss
    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        recon_x = self.decoder_x(x, edge_index)
        return recon_x



parser = argparse.ArgumentParser(description="Train a VGAE model on different datasets.")
parser.add_argument('--dataset', type=str, default='german', choices=['german', 'credit','nba'],
                    help='The name of the dataset to use.')
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--latent_dim', type=int, default=16)
parser.add_argument('--train_percentage', type=float, default=0.7)
parser.add_argument('--hgr', type=str, default="use",choices = ['use','no'])
parser.add_argument('--epoch', type=int, default=1000)
args = parser.parse_args()

s_hat_index = None
GAE_model = None
num_features = None
# 根据命令行参数选择数据集
if args.dataset == 'german':
    dataset_class = GermanDataset
    num_features = 26
    s_hat_index = torch.Tensor([0, 1, 2])
elif args.dataset == 'credit':
    dataset_class = CreditDataset
    num_features = 12
    s_hat_index = torch.Tensor([0])
elif args.dataset == 'nba':
    dataset_class = NbaDataset
    num_features = 96
    s_hat_index = torch.Tensor([0])
else:
    raise ValueError("Unsupported dataset.")

# 加载数据集
dataset = dataset_class(root='./')
data = dataset[0]

num_nodes = data.num_nodes
train_percentage = args.train_percentage
num_train_nodes = int(train_percentage * num_nodes)
hgr = 'use_hgr' if args.hgr == 'use' else 'no_hgr'
# Create a boolean mask for train mask
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[: num_train_nodes] = True

# Add train mask to data object
data.train_mask = train_mask

# Create a boolean mask for test mask
test_mask = ~data.train_mask
data.test_mask = test_mask

input_dim = output_dim = num_features
gender_dim = 2
model = GumbelSoftmaxVAE(input_dim, args.hidden_dim, args.latent_dim, output_dim, gender_dim)

GAE_model = GAE(len(s_hat_index), args.hidden_dim, args.latent_dim, len(s_hat_index))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
optimizer_s = torch.optim.Adam(GAE_model.parameters(), lr=0.001, weight_decay=1e-4)

epoch = args.epoch
best_loss = float('inf')
best_gender_acc = 0.0

_,_, labels,sens = load_data('/home/chuzhibo/fairGNN-WOD/', args.dataset)
labels = labels.to(device)
sens = sens.to(device)

logger = logging.getLogger('training_logger')
logger.setLevel(logging.DEBUG)  # 设置最低日志级别

# 创建一个handler用于写入日志文件
fh = logging.FileHandler(f"./log/{args.dataset}_logfile.log")
fh.setLevel(logging.DEBUG)  # 设置handler的日志级别

# 定义handler的输出格式
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)

# 测试日志输出
logger.info('Testing started.')
GAE_model.to(device)


def train():
    global s_hat_index
    model.train()
    GAE_model.train()
    
    optimizer.zero_grad()
    optimizer_s.zero_grad()  # Clear gradients
    # print(type(data.x),type(data.edge_index))
    x_hat, gender_hat,logits_label, mu, logvar, z = model(data.x, data.edge_index)
    s_hat_index = s_hat_index.clone().detach()
    s_hat_index = s_hat_index.to(torch.long)
    # s_hat_index = torch.tensor(s_hat_index, dtype=torch.long)
    # s_hat_index = s_hat_index.to(device)
    x_hat_s = GAE_model(data.x[:,s_hat_index], data.edge_index)
    


    data.train_mask = data.train_mask.to(device)
    recon_x_loss = F.mse_loss(x_hat[data.train_mask], data.x[data.train_mask], reduction='sum')
    recon_x_loss_s = F.mse_loss(x_hat_s[data.train_mask], data.x[data.train_mask][:,s_hat_index], reduction='sum')

    gender_probabilities = gender_hat[torch.arange(gender_hat.shape[0]), 0]
    recon_edge_loss = model.recon_edge_loss(z, data.edge_index)

    recon_loss = recon_x_loss + recon_edge_loss
    criterion1 = nn.BCELoss()
    gender_probabilities = gender_probabilities[data.train_mask].to(device)
    
    gender_recon_loss = criterion1(gender_probabilities, sens[data.train_mask].float())


    
    criterion2 = torch.nn.BCEWithLogitsLoss()
    logits_label = logits_label[data.train_mask].to(device)
    label_loss = criterion2(logits_label.squeeze(dim=1), labels[data.train_mask].float())

    if hgr == 'hgr':
        s_hat = GAE_model.encode(data.x[:,s_hat_index], data.edge_index)
        loss = recon_loss + (1 / data.num_nodes) * model.kl_loss(mu, logvar) + gender_recon_loss + label_loss + hgr_correlation(z, s_hat)
    else:
        loss = recon_loss + (1 / data.num_nodes) * model.kl_loss(mu, logvar) + gender_recon_loss + label_loss 

    loss.backward()  # Derive gradients
    optimizer.step()  # Update parameters

    loss_s = recon_x_loss_s
    loss_s.backward()  # Derive gradients
    optimizer_s.step()  # Update parameters

    return loss,loss_s

max_acc = -np.inf
max_f1 = -np.inf
max_spd = -np.inf
max_eod = -np.inf
max_auc = -np.inf
max_gen = -np.inf
def test():
    global max_acc, max_f1,max_spd,max_eod,max_auc,max_gen
    model.eval()
    _, gender_hat, label_hat, _,_,_ = model(data.x, data.edge_index)

    sens_reshaped = sens.to(device).float()
    gender_pred = [0 if g[0] == 0 else 1 for g in gender_hat]
    gender_pred = torch.Tensor(gender_pred).to(device)
    # print("gender_pred: ", gender_pred[data.test_mask])
    # print("---------------------------")
    # print("sens_reshaped: ", sens_reshaped[data.test_mask])
    gender_correct = gender_pred[data.test_mask] == sens_reshaped[data.test_mask]  # Count correct predictions
    logger.info(f"gender_pred:  model gender 0: {int((gender_pred[data.test_mask] == 0).sum())}, ground true gender 0: {int((sens_reshaped[data.test_mask] == 0).sum())}")
    logger.info(f"gender_pred:  model gender 1: {int((gender_pred[data.test_mask] == 1).sum())}, ground true gender 1: {int((sens_reshaped[data.test_mask] == 1).sum())}")
    # print(gender_pred)

    # print(f"gender_correct.sum() :{gender_correct.sum()},  int(data.test_mask.sum()): {int(data.test_mask.sum())}")
    
    label_hat = label_hat.squeeze()
    label_probabilities = torch.sigmoid(label_hat)

    label_predictions = (label_probabilities >= 0.5).float()
    logger.info(f"label_predictions:  model label 0: {int((label_predictions[data.test_mask] == 0).sum())}, ground true label 0: {int((labels[data.test_mask] == 0).sum())}")
    logger.info(f"label_predictions:  model label 1: {int((label_predictions[data.test_mask] == 1).sum())}, ground true label 1: {int((labels[data.test_mask] == 1).sum())}")


    label_correct = label_predictions[data.test_mask] == labels[data.test_mask]  # Count correct predictions
    # print(f"label_correct percentage :{int(label_correct.sum())/ int(data.test_mask.sum())},  int(data.test_mask.sum()): {int(data.test_mask.sum())}")
    gender_correct = int(gender_correct.sum()) / int(data.test_mask.sum())  # Get proportion of correct predictions
    label_correct = int(label_correct.sum())/ int(data.test_mask.sum())
    from metric import metric
    acc, f1, spd, aod, eod, recall, auc = metric(label_predictions[data.test_mask], labels[data.test_mask], sens[data.test_mask])
    max_acc = max(max_acc, acc)
    max_f1 = max(max_f1, f1)
    max_spd = max(max_spd, spd)
    max_eod = max(max_eod, eod)
    max_auc = max(max_auc, auc)
    max_gen = max(gender_correct,max_gen)
    print(max_acc, max_f1, max_spd, max_eod, max_auc,max_gen)
    return gender_correct, acc, f1, spd, aod, eod, recall, auc

if __name__ == "__main__":
    writer = SummaryWriter('runs/accuracy_experiment')
    cur = 0
    while True:
        if epoch == cur:
            break
        loss,loss_s = train()

        gender_acc, acc, f1, spd, aod, eod, recall, auc = test()
        logger.info(f"Epoch {cur}, gender acc: {gender_acc}, label acc: {acc}, f1: {f1}, auc: {auc}, spd: {spd}, eod: {eod}")
        logger.info("---------------------------")
        # writer.add_scalar('Accuracy/train', acc)
        if gender_acc > best_gender_acc:
            best_gender_acc = gender_acc
            checkpoint_path = f"/home/chuzhibo/fairGNN-WOD/modelpth/{hgr}/"+args.dataset +"_best_model.pth"
            print(checkpoint_path)
            torch.save({
                'epoch': cur,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
        cur += 1
    writer.close()
    print('Training completed.')