import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_mean, scatter_sum, scatter_softmax
from logging import getLogger
from scipy.sparse import csr_matrix
import scipy.sparse as sp

def wasserstein_1d_torch(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute 1D Earth Mover’s Distance (Wasserstein) between two non-negative
    vectors p, q of the same length (treating them as discrete distributions).
    """
    # ensure non-negative
    p = p.clamp(min=0)
    q = q.clamp(min=0)
    # normalize to sum=1
    p = p / (p.sum() + 1e-8)
    q = q / (q.sum() + 1e-8)
    # cumulative sums (CDFs)
    cdf_p = torch.cumsum(p, dim=0)
    cdf_q = torch.cumsum(q, dim=0)
    # EMD = L1 distance between CDFs
    return torch.sum(torch.abs(cdf_p - cdf_q))


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

def _edge_sampling(edge_index, edge_type, rate=0.5):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices]


def _sparse_dropout(x, rate=0.5):
    noise_shape = x._nnz()

    random_tensor = torch.rand(noise_shape).to(x.device) + rate
    dropout_mask = torch.floor(random_tensor).type(torch.bool)
    i = x._indices()
    v = x._values()

    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse_coo_tensor(
        i,
        v,
        x.shape,  # sparse 張量的 size
        dtype=v.dtype,
        device=x.device
    )
    return out * (1. / (1 - rate))


class RGAT(nn.Module):
    def __init__(self, channel, n_hops,
                 mess_dropout_rate=0.4):
        super(RGAT, self).__init__()
        self.mess_dropout_rate = mess_dropout_rate
        self.W = nn.Parameter(torch.empty(size=(channel, channel)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * channel, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Linear(2 * channel, channel)

        self.leakyrelu = nn.LeakyReLU(0.2)

        self.n_hops = n_hops

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    ## aggregate kg info
    def agg(self, entity_emb, relation_emb, kg):
        edge_index, edge_type = kg
        head, tail = edge_index
        a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)
        e_input = torch.multiply(self.fc(a_input), relation_emb[edge_type]).sum(-1)  # N,e
        e = self.leakyrelu(e_input)  # (N, e_num)
        e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])
        agg_emb = entity_emb[tail] * e.view(-1, 1)
        agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])
        agg_emb = agg_emb + entity_emb
        return agg_emb

    def forward(self, entity_emb, relation_emb, kg, mess_dropout=True):
        entity_res_emb = entity_emb
        for _ in range(self.n_hops):
            entity_emb = self.agg(entity_emb, relation_emb, kg)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
        return entity_emb


class KGELSA(nn.Module):
    def __init__(self, data_config, args_config, kg_graph, adj_mat):
        super(KGELSA, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        self.logger = getLogger()
        self.args = args_config

        self.tau = args_config.tau
        self.cl_weight = args_config.cl_weight
        self.mu = args_config.mu

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.sim_metric = args_config.sim_metric
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")
        self.ui_mat = adj_mat
        self.binorm_adj = self._make_binorm_adj(adj_mat)
        self.edge_index, self.edge_type = self._get_edges(kg_graph)
        ## trainerable parameters
        self.all_embed = nn.init.xavier_uniform_(torch.empty(self.n_nodes, self.emb_size))
        self.relation_embed = nn.init.xavier_uniform_(torch.empty(self.n_relations, self.emb_size))
        self.all_embed = nn.Parameter(self.all_embed)
        self.relation_embed = nn.Parameter(self.relation_embed)
        self.rgat = RGAT(self.emb_size, self.context_hops, self.mess_dropout_rate)

        self.elsa_scores = None
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.align_weight = args_config.align_weight

    def _make_binorm_adj(self, mat):
        a = csr_matrix((self.n_users, self.n_users))
        b = csr_matrix((self.n_items, self.n_items))
        mat = sp.vstack(
            [sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0# MARK
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        mat = mat.dot(d_inv_sqrt_mat).transpose().dot(
            d_inv_sqrt_mat).tocoo()

        # make torch tensor
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse_coo_tensor(
            idxs,
            vals,
            shape,
            dtype=vals.dtype,
            device=self.device
        )

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(
            i,  # indices: shape [ndim, nnz]
            v,  # values: shape [nnz]
            coo.shape,  # sparse 張量的大小
            dtype=v.dtype,  # 跟原本的 values 保持相同 dtype
            device=coo.device  # 或 self.device
        )

    def get_ui_aug_views(self, kg_stability, mu):
        kg_stability = torch.exp(kg_stability)
        kg_weights = (kg_stability - kg_stability.min()) / (kg_stability.max() - kg_stability.min())
        kg_weights = kg_weights.where(kg_weights > 0.3, torch.ones_like(kg_weights) * 0.3)

        # overall probability of keep
        weights = mu / torch.mean(kg_weights) * (kg_weights)
        weights = weights.where(weights < 0.95, torch.ones_like(weights) * 0.95)

        items_in_edges = self.ui_mat.col
        edge_weights = weights[items_in_edges]
        edge_mask = torch.bernoulli(edge_weights).to(torch.bool).cpu()
        keep_rate = edge_mask.sum().item() / edge_mask.size()[0]
        # print(f"u-i edge keep ratio: {keep_rate:.2f}")
        # drop
        col = self.ui_mat.col
        row = self.ui_mat.row
        v = self.ui_mat.data

        col = col[edge_mask]
        row = row[edge_mask]
        v = v[edge_mask]

        masked_ui_mat = sp.coo_matrix((v, (row, col)), shape=(self.n_users, self.n_items))
        return self._make_binorm_adj(masked_ui_mat)

    @torch.no_grad()
    def get_aug_views(self):
        entity_emb = self.all_embed[self.n_users:, :]
        kg_view_1 = _edge_sampling(self.edge_index, self.edge_type, self.node_dropout_rate)
        kg_view_2 = _edge_sampling(self.edge_index, self.edge_type, self.node_dropout_rate)
        kg_v1_ro = self.rgat(entity_emb, self.relation_embed, kg_view_1, False)[:self.n_items, :]
        kg_v2_ro = self.rgat(entity_emb, self.relation_embed, kg_view_2, False)[:self.n_items, :]

        # 4. 根據 sim_metric 選擇相似度計算
        if self.sim_metric == 'wasserstein':
            # 用 1/(1 + EMD) 作為穩定性分數
            dists = [wasserstein_1d_torch(kg_v1_ro[i], kg_v2_ro[i])
                     for i in range(self.n_items)]
            dists = torch.stack(dists, dim=0).to(self.device)
            stability = 1.0 / (1.0 + dists)
        else:
            # 默認用 cosine similarity
            stability = F.cosine_similarity(kg_v1_ro, kg_v2_ro, dim=-1)

        ui_view_1 = self.get_ui_aug_views(stability, mu=self.mu)
        ui_view_2 = self.get_ui_aug_views(stability, mu=self.mu)
        return kg_view_1, kg_view_2, ui_view_1, ui_view_2

    def forward(self, batch):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        kg_view_1, kg_view_2, ui_view_1, ui_view_2 = batch['aug_views']

        if self.node_dropout:
            g_droped = _sparse_dropout(self.binorm_adj, self.node_dropout_rate)
            edge_index, edge_type = _edge_sampling(self.edge_index, self.edge_type, self.node_dropout_rate)
        else:
            g_droped = self.binorm_adj
            edge_index, edge_type = self.edge_index, self.edge_type

        user_emb, item_emb = self.gcn(edge_index, edge_type, g_droped)
        u_e = user_emb[user]
        pos_e, neg_e = item_emb[pos_item], item_emb[neg_item]

        # 1) 原始分數
        pos_scores_kg = torch.sum(u_e * pos_e, dim=1)  # [B]
        neg_scores_kg = torch.sum(u_e * neg_e, dim=1)  # [B]

        pos_scores = pos_scores_kg
        neg_scores = neg_scores_kg

        # 4) 用 softplus 計算 BPR loss
        rec_loss = torch.sum(F.softplus(-(pos_scores - neg_scores)))

        # 5) 一樣保留正則項
        reg_loss = (1 / 2) * (
                u_e.norm(2).pow(2) +
                pos_e.norm(2).pow(2) +
                neg_e.norm(2).pow(2)
        ) / float(u_e.shape[0])

        # CL
        users_v1_ro, items_v1_ro = self.gcn(kg_view_1[0], kg_view_1[1], ui_view_1)
        users_v2_ro, items_v2_ro = self.gcn(kg_view_2[0], kg_view_2[1], ui_view_2)
        user_cl_loss = self.infonce_overall(users_v1_ro[user], users_v2_ro[user], users_v2_ro)
        item_cl_loss = self.infonce_overall(items_v1_ro[pos_item], items_v2_ro[pos_item], items_v2_ro)

        cl_loss = self.cl_weight * (user_cl_loss + item_cl_loss)
        loss = rec_loss + self.decay * reg_loss + cl_loss

        # loss = rec_loss + self.decay * reg_loss + cl_loss + self.align_weight * align_loss

        loss_dict = {
            "rec_loss": rec_loss.item(),
            "cl_loss": cl_loss.item(),
            # "align_loss" : align_loss.item()
        }
        return loss, loss_dict

    def gcn(self, edge_index, edge_type, g):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        ## agg
        entity_emb = self.rgat(entity_emb, self.relation_embed, [edge_index, edge_type],
                               self.mess_dropout & self.training)[:self.n_items, :]

        res_emb = [torch.cat([user_emb, entity_emb], dim=0)]
        for l in range(self.context_hops):  # n hops
            all_emb = torch.sparse.mm(g, res_emb[-1])
            res_emb.append(all_emb)
        res_emb = sum(res_emb)
        user_res_emb, item_res_emb = res_emb.split([self.n_users, self.n_items], dim=0)
        return user_res_emb, item_res_emb

    def generate(self):
        return self.gcn(self.edge_index, self.edge_type, self.binorm_adj)

    def calc_kg_loss_transE(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        entity_emb = self.all_embed[self.n_users:, :]
        r_embed = self.relation_embed[r]  # (kg_batch_size, relation_dim)
        h_embed = entity_emb[h]  # (kg_batch_size, entity_dim)
        pos_t_embed = entity_emb[pos_t]  # (kg_batch_size, entity_dim)
        neg_t_embed = entity_emb[neg_t]  # (kg_batch_size, entity_dim)
        # Equation (1)
        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)  # (kg_batch_size)
        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(
            neg_t_embed)
        loss = kg_loss + 1e-3 * l2_loss
        return loss

    def infonce_overall(self, z1, z2, z_all):

        def sim(z1: torch.Tensor, z2: torch.Tensor):
            if z1.size()[0] == z2.size()[0]:
                return F.cosine_similarity(z1, z2)
            else:
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
                return torch.mm(z1, z2.t())

        f = lambda x: torch.exp(x / self.tau)
        # batch_size
        between_sim = f(sim(z1, z2))
        # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
        all_sim = f(sim(z1, z_all))
        # batch_size
        positive_pairs = between_sim
        # batch_size
        negative_pairs = torch.sum(all_sim, 1)
        loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
        return loss

    def scores(self, user_indices, item_indices, u_g_embeddings, i_g_embeddings):
        # 1) 確保 indices 是 Tensor
        device = u_g_embeddings.device
        if isinstance(user_indices, list):
            user_indices = torch.tensor(user_indices, dtype=torch.long, device=device)
        if isinstance(item_indices, list):
            item_indices = torch.tensor(item_indices, dtype=torch.long, device=device)

        # 2) 計算 KG 內積 → [B, I_batch]
        kg_scores = u_g_embeddings @ i_g_embeddings.t()
        # mod = self.alpha * torch.sigmoid(kg_scores)

        # 3) 從 self.ease_scores 取出對應子矩陣 → [B, I_batch]
        #    先按 user_indices 選行，再按 item_indices 選列
        ui_cpu = user_indices.cpu()
        ii_cpu = item_indices.cpu()
        # ease_scores = self.ease_scores[user_indices][:, item_indices]
        elsa_batch_cpu = self.elsa_scores[ui_cpu][:, ii_cpu]
        elsa_batch = elsa_batch_cpu.to(device, non_blocking=True)
        # final_score = ease_batch + mod
        # return final_score
        # return kg_scores * ease_scores
        return kg_scores * elsa_batch
        # return ease_batch

    def print_shapes(self):
        self.logger.info("########## Model HPs ##########")
        self.logger.info("tau: {}".format(self.tau))
        self.logger.info("cl_weight: {}".format(self.cl_weight))
        self.logger.info("mu: {}".format(self.mu))
        self.logger.info("hard_update_interval: {}".format(self.args.hard_update_interval))
        self.logger.info("hard_k: {}".format(self.args.hard_k))
        self.logger.info("hard_p: {:.2f}".format(self.args.hard_p))
        self.logger.info("########## Model Parameters ##########")
        self.logger.info("learning rate: %.4f", self.args.lr)
        self.logger.info("context_hops: %d", self.context_hops)
        self.logger.info("node_dropout: %d", self.node_dropout)
        self.logger.info("node_dropout_rate: %.1f", self.node_dropout_rate)
        self.logger.info("mess_dropout: %d", self.mess_dropout)
        self.logger.info("mess_dropout_rate: %.1f", self.mess_dropout_rate)
        self.logger.info('all_embed: {}'.format(self.all_embed.shape))
        self.logger.info('interact_mat: {}'.format(self.binorm_adj.shape))
        self.logger.info('edge_index: {}'.format(self.edge_index.shape))
        self.logger.info('edge_type: {}'.format(self.edge_type.shape))
        self.logger.info('sim_metric: {}'.format(self.args.sim_metric))
# ---------------------------
# 輔助函數：將 train_cf 轉換為 user-item 交互矩陣
def build_user_item_matrix(train_cf, n_users, n_items, device):
    """
    將 train_cf (shape: [N, 2]) 轉換成 shape: (n_users, n_items)
    每列代表一個使用者對所有項目的交互（此處以 0/1 編碼）
    """
    mat = np.zeros((n_users, n_items), dtype=np.float32)
    for u, i in train_cf:
        mat[int(u), int(i)] = 1.0
    return torch.tensor(mat, device=device)


