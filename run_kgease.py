
import os
import random
import numpy as np
import torch
from time import time
from collections import defaultdict
from tqdm import tqdm
from prettytable import PrettyTable
from logging import getLogger
from utils.parser import parse_args_kgease
from utils.data_loader import load_data
from modules.KGVAE.kgease import KGEASE
from utils.evaluator import Evaluator
from utils.helper import early_stopping, init_logger
from utils.sampler import UniformSampler
from modules.KGVAE.kgease import build_user_item_matrix
from modules.KGVAE.ease import EASE
from scipy.sparse import csr_matrix

def neg_sampling(cf_pairs, train_user_dict, hard_neg_set):
    t1 = time()
    cf_negs = []
    for u, _ in cf_pairs:
        if random.random() < args.hard_p and hard_neg_set.get(u):
            neg = random.choice(hard_neg_set[u])
        else:
            # 這裡先傳入 np.array([u])，再索引 [0,0] 才是真正的 scalar
            neg_array = sampling.sample_negative(np.array([u]), n_items, train_user_dict, 1)  # shape (1,1)
            neg = int(neg_array[0, 0])
        cf_negs.append(neg)

    cf_triples = np.concatenate([cf_pairs, np.array(cf_negs)[:, None]], axis=1)
    t2 = time()
    logger.info("Negative sampling time: %.2fs", t2 - t1)
    logger.info("train_cf_triples shape: %s", str(cf_triples.shape))
    return cf_triples

#
# def neg_sampling(cf_pairs, train_user_dict):
#     t1 = time()
#     cf_negs = sampling.sample_negative(cf_pairs[:, 0], n_items, train_user_dict, 1)
#     cf_triples = np.concatenate([cf_pairs, cf_negs], axis=1)
#     t2 = time()
#     logger.info("Negative sampling time: %.2fs", t2 - t1)
#     logger.info("train_cf_triples shape: %s", str(cf_triples.shape))
#     return cf_triples

def get_feed_dict(data, start, end):
            feed = {}
            pairs = torch.from_numpy(data[start:end]).to(device).long()
            feed['users'] = pairs[:, 0]
            feed['pos_items'] = pairs[:, 1]
            feed['neg_items'] = pairs[:, 2]
            feed['batch_start'] = start
            return feed
# ---------------------------
# 主訓練流程：分為預訓練 RecVAE 和 KGCL 兩階段
if __name__ == '__main__':
    # 固定隨機種子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from setproctitle import setproctitle

    setproctitle('EXP@KGEASE')
    sampling = UniformSampler(seed)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)
    logger = getLogger()
    try:
        # 讀取參數與初始化 logger
        args = parse_args_kgease()
        device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
        log_fn = init_logger(args)

        logger.info("PID: %d", os.getpid())
        logger.info("Experiment Description: %s", args.desc)

        # 讀取資料
        train_cf, test_cf, user_dict, n_params, graph, kg_dict, adj_mat = load_data(args)
        n_users = n_params['n_users']
        n_items = n_params['n_items']

        # 建立 user-item 交互矩陣 (用於 RecVAE 輸入)
        user_item_matrix = build_user_item_matrix(train_cf, n_users, n_items, 'cpu')

        # 建立 KGEASE 模型
        model = KGEASE(n_params, args, graph, adj_mat).to(device)
        model.print_shapes()
        print(">>> data_config['n_items'] :", n_items)
        print(">>> model.n_items          :", model.n_items)
        # ────────────────────────────────────────────────────────────
        # 1) 初始化並 fit
        # ease_model = EASE(reg=args.ease_reg).to(device)
        #
        # print(">>> user_item_matrix.shape:", user_item_matrix.shape)
        # ease_model.fit(user_item_matrix)  # user_item_matrix shape = [n_users, n_items]
        #
        # # 2) 一次性算出所有 user×item 的分數
        # ease_scores = ease_model(user_item_matrix)  # [n_users, n_items]
        # print(">>> ease_scores.shape:", ease_scores.shape)
        #
        # model.ease_scores = ease_scores

        R_sparse = csr_matrix(user_item_matrix)
        ease_model = EASE(reg=args.ease_reg).to('cpu')
        ease_model.fit(R_sparse)

        R_dense = torch.tensor(R_sparse.toarray(), dtype=torch.float32)  # or test data
        ease_scores = ease_model(R_dense)  # [n_users, n_items] on CPU
        print(">>> ease_scores.shape:", ease_scores.shape)
        model.ease_scores = ease_scores.half().to(device)

        # ─────────────────────────────────────
        # 第二階段：訓練 KGCL 部分（包括 CF 與 KG 損失）
        rec_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        kg_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        ## scheduler
        rec_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            rec_optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        kg_scheduler = torch.optim.lr_scheduler.StepLR(kg_optimizer, step_size=10, gamma=0.9)

        evaluator = Evaluator(args)
        test_interval = 1
        early_stop_step = 10
        cur_best_pre_0 = 0
        cur_stopping_step = 0

        logger.info("Starting training...")
        cur_epoch = 0

        hard_neg_dict = {u: [] for u in range(n_users)}

        for epoch in range(args.epoch):

            ##sample top-k hardest
            if epoch % args.hard_update_interval == 0:
                model.eval()
                with torch.no_grad():
                    # 拿到全量 embedding 和 ease_scores
                    u_emb, i_emb = model.generate()  # [n_users, D], [n_items, D]
                    ease_gpu = model.ease_scores.to(device, non_blocking=True)

                    for u in range(n_users):
                        # 1) 計算該 user 的全量分數、排除訓練正樣本
                        scores = (u_emb[u] @ i_emb.t()) * ease_gpu[u]  # [n_items]
                        pos = set(user_dict['train_user_set'][u])
                        scores[list(pos)] = -1e9

                        # 2) 取 xx–xx 百分位作為 semi-hard 的範圍
                        vals = scores.cpu().numpy()
                        low, high = np.percentile(vals, [args.lower_bound, args.upper_bound])
                        mask = (vals >= low) & (vals <= high)

                        # 3) 將符合條件的 index 作為候選
                        hard_neg_dict[u] = np.where(mask)[0].tolist()

            cur_epoch = epoch
            # ----- CF 部分訓練 -----
            # ----- cf data ----- """
            # train_cf_with_neg = neg_sampling(train_cf, user_dict['train_user_set'])

            ## hard sampeling
            train_cf_with_neg = neg_sampling(train_cf, user_dict['train_user_set'], hard_neg_set=hard_neg_dict)
            indices = np.arange(len(train_cf))
            np.random.shuffle(indices)
            train_cf_with_neg = train_cf_with_neg[indices]

            # ----- training cf-----"""
            model.train()

            # -----  ----- -----
            aug_views = model.get_aug_views()
            add_loss_dict = defaultdict(float)
            s = 0
            train_start = time()
            batch_num = len(train_cf) // args.batch_size
            for _ in tqdm(range(batch_num), desc=f"Epoch {epoch} CF Training"):

                batch = get_feed_dict(train_cf_with_neg, s, s + args.batch_size)
                # 產生 augmentation views
                batch['aug_views'] = aug_views

                batch_loss, batch_loss_dict = model(batch)
                rec_optimizer.zero_grad(set_to_none=True)
                # optimizer_kg.zero_grad()
                batch_loss.backward()
                rec_optimizer.step()
                # optimizer_kg.step()

                for k, v in batch_loss_dict.items():
                    add_loss_dict[k] += v / len(train_cf)
                s += args.batch_size
            cf_time = time() - train_start

            # ----- KG 部分訓練 -----
            kg_start = time()
            kg_total_loss = 0
            n_kg_batch = n_params['n_triplets'] // 4096
            from utils.data_loader import generate_kg_batch

            for _ in tqdm(range(1, n_kg_batch + 1), desc=f"Epoch {epoch} KG Training"):
                kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = generate_kg_batch(
                    kg_dict, 4096, n_params['n_entities']
                )
                kg_batch_head = kg_batch_head.to(device)
                kg_batch_relation = kg_batch_relation.to(device)
                kg_batch_pos_tail = kg_batch_pos_tail.to(device)
                kg_batch_neg_tail = kg_batch_neg_tail.to(device)
                kg_loss = model.calc_kg_loss_transE(kg_batch_head, kg_batch_relation,
                                                         kg_batch_pos_tail, kg_batch_neg_tail)
                kg_optimizer.zero_grad()
                kg_loss.backward()
                kg_optimizer.step()
                kg_total_loss += kg_loss.item()
            kg_time = time() - kg_start

            logger.info("Epoch %04d | CF Time: %.1fs | KG Time: %.1fs | Mean KG Loss: %.4f",
                        epoch, cf_time, kg_time, kg_total_loss / n_kg_batch)

            kg_scheduler.step()

            # ----- 評估 -----
            if epoch % test_interval == 0 and epoch >= 0:
                model.eval()
                test_start = time()
                logger.info("Mixture ratio for testing: %04f ", model.alpha)
                with torch.no_grad():
                    ret = evaluator.test(model, user_dict, n_params)
                test_time = time() - test_start
                results = PrettyTable()
                results.field_names = ["Epoch", "CF Time", "KG Time", "Recall", "NDCG", "Precision", "Hit Ratio"]
                results.add_row([epoch, f"{cf_time:.1f}", f"{kg_time:.1f}",
                                 ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']])
                logger.info("\n" + str(results))

                ## update scheduler
                recall_at_20 = ret['recall'][0]
                rec_scheduler.step(recall_at_20)
                ##
                cur_best_pre_0, cur_stopping_step, should_stop = early_stopping(
                    ret['recall'][0], cur_best_pre_0, cur_stopping_step,
                    expected_order='acc', flag_step=early_stop_step)
                if cur_stopping_step == 0:
                    logger.info("### Found better model at epoch %d", epoch)
                elif should_stop:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        logger.info("Training complete. Early stopping at epoch %d, recall@20: %.4f", cur_epoch, cur_best_pre_0)

    except Exception as e:
        logger.exception(e)