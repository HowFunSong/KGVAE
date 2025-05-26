
import os
import random
import numpy as np
import torch
from time import time
from collections import defaultdict
from tqdm import tqdm
from prettytable import PrettyTable
from logging import getLogger
from utils.parser import parse_args_kgvae_enc
from utils.data_loader import load_data
from modules.KGVAE.kgvae_enc import KGVAEENC
from utils.evaluator import Evaluator
from utils.helper import early_stopping, init_logger
from utils.sampler import UniformSampler
from modules.KGVAE.kgvae import build_user_item_matrix
from modules.KGVAE.vae import RecVAETrainer
def neg_sampling(cf_pairs, train_user_dict):
    t1 = time()
    cf_negs = sampling.sample_negative(cf_pairs[:, 0], n_items, train_user_dict, 1)
    cf_triples = np.concatenate([cf_pairs, cf_negs], axis=1)
    t2 = time()
    logger.info("Negative sampling time: %.2fs", t2 - t1)
    logger.info("train_cf_triples shape: %s", str(cf_triples.shape))
    return cf_triples

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

    setproctitle('EXP@KGVAE')
    sampling = UniformSampler(seed)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)
    logger = getLogger()
    try:
        # 讀取參數與初始化 logger
        args = parse_args_kgvae_enc()
        device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
        log_fn = init_logger(args)

        logger.info("PID: %d", os.getpid())
        logger.info("Experiment Description: %s", args.desc)

        # 讀取資料
        train_cf, test_cf, user_dict, n_params, graph, kg_dict, adj_mat = load_data(args)
        n_users = n_params['n_users']
        n_items = n_params['n_items']

        # 建立 KGVAE 模型
        model = KGVAEENC(n_params, args, graph, adj_mat).to(device)
        model.print_shapes()

        recvae_trainer = RecVAETrainer(
            recvae=model.recvae,
            user_embeds=torch.zeros(n_users, args.dim, device=device),
            lr=args.lr * 0.1,
            n_enc_epochs=3,
            n_dec_epochs=1,
            device=device,
        )
        # ─────────────────────────────────────
        # 第二階段：訓練 KGCL 部分（包括 CF 與 KG 損失）
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        evaluator = Evaluator(args)
        test_interval = 1
        early_stop_step = 5
        cur_best_pre_0 = 0
        cur_stopping_step = 0

        logger.info("Starting KGCL training...")
        cur_epoch = 0
        for epoch in range(args.epoch):
            cur_epoch = epoch
            # ----- CF 部分訓練 -----
            # ----- cf data ----- """
            train_cf_with_neg = neg_sampling(train_cf, user_dict['train_user_set'])
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
                # KGCL 內部產生 augmentation views
                batch['aug_views'] = aug_views
                batch_loss, batch_loss_dict = model(batch)
                # 歸零梯度
                optimizer.zero_grad()
                ## 反向傳播
                batch_loss.backward()
                ## 更新
                optimizer.step()

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
                optimizer.zero_grad()
                kg_loss.backward()
                optimizer.step()
                kg_total_loss += kg_loss.item()
            kg_time = time() - kg_start

            logger.info("Epoch %04d | CF Time: %.1fs | KG Time: %.1fs | Mean KG Loss: %.4f",
                        epoch, cf_time, kg_time, kg_total_loss / n_kg_batch)
            # ----- vae 訓練 -----+
            # --- VAE 訓練階段 ---
            model.eval()
            with torch.no_grad():
                user_emb, _ = model.generate()
            recvae_trainer.R = user_emb
            avg_enc, avg_dec = recvae_trainer.train_one_epoch(dropout_rate=0.5)
            logger.info(
                "Epoch %04d | RecVAE EncAvgLoss: %.4f | DecAvgLoss: %.4f",
                epoch, avg_enc, avg_dec
            )

            # ----- 評估 -----
            if epoch % test_interval == 0 and epoch >= 0:
                model.eval()
                model.recvae.eval()
                test_start = time()
                with torch.no_grad():
                    ret = evaluator.test(model, user_dict, n_params)
                test_time = time() - test_start
                results = PrettyTable()
                results.field_names = ["Epoch", "CF Time", "KG Time", "Recall", "NDCG", "Precision", "Hit Ratio"]
                results.add_row([epoch, f"{cf_time:.1f}", f"{kg_time:.1f}",
                                 ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']])
                logger.info("\n" + str(results))

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