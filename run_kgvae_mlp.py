
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from utils.plotter import Plotter
from time import time
from collections import defaultdict
from tqdm import tqdm
from prettytable import PrettyTable
from logging import getLogger
from utils.parser import parse_args_kgvae_mlp
from utils.data_loader import load_data
from modules.KGVAE.kgvae_mlp import KGVAEMLP
from utils.evaluator import Evaluator
from utils.helper import early_stopping, init_logger
from utils.sampler import UniformSampler
from modules.KGVAE.kgvae_mlp import build_user_item_matrix
from modules.KGVAE.vae import VAE
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

    setproctitle('EXP@KGVAE_MLP')
    sampling = UniformSampler(seed)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)
    logger = getLogger()
    try:
        # 讀取參數與初始化 logger
        args = parse_args_kgvae_mlp()
        device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
        log_fn = init_logger(args)

        logger.info("PID: %d", os.getpid())
        logger.info("Experiment Description: %s", args.desc)

        plot_dir = os.path.join("plots", args.model, args.dataset)
        os.makedirs(plot_dir, exist_ok=True)
        plotter = Plotter(out_dir=plot_dir)

        # 讀取資料
        train_cf, test_cf, user_dict, n_params, graph, kg_dict, adj_mat = load_data(args)
        n_users = n_params['n_users']
        n_items = n_params['n_items']

        # 建立 user-item 交互矩陣 (用於 RecVAE 輸入)
        user_item_matrix = build_user_item_matrix(train_cf, n_users, n_items, device)

        recvae = VAE(hidden_dim=args.dim, latent_dim=args.dim, input_dim=n_items)
        # ────────────────────────────────────────────────────────────
        # 第一階段：用 RecVAETrainer 預訓練 RecVAE（交替 Encoder/Decoder）
        recvae_trainer = RecVAETrainer(
            recvae=recvae,
            user_embeds=user_item_matrix,
            lr=args.lr,
            n_enc_epochs=args.n_enc_epochs,
            n_dec_epochs=args.n_dec_epochs,
            device=device
        )
        logger.info("Pretraining RecVAE for %d epochs (enc %d / dec %d)...",
                    args.vae_epochs, args.n_enc_epochs, args.n_dec_epochs)
        final_enc, final_dec = recvae_trainer.pretrain(args.vae_epochs)
        logger.info("RecVAE pretraining complete. Final encoder loss=%.4f, decoder loss=%.4f",
                    final_enc, final_dec)

        logger.info("Pretraining of RecVAE complete")
        recvae.eval()
        with torch.no_grad():
            mu, _ = recvae.encoder(
                user_item_matrix,
                dropout_rate=0
            )
        mu.detach()

        recvae.to('cpu')

        del recvae
        del recvae_trainer
        torch.cuda.empty_cache()

        # ─────────────────────────────────────
        # 第二階段：訓練 KGCL 部分（包括 CF 與 KG 損失）
        # 建立 KGVAE 模型
        model = KGVAEMLP(n_params, args, graph, adj_mat, user_item_matrix).to(device)
        model.kgcl.print_shapes()
        model.print_hyper_parameters()
        # print(mu)
        model.cache_user_latent(mu)

        optimizer_kg = torch.optim.Adam(model.kgcl.parameters(), lr=args.lr)
        optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr)

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
            aug_views = model.kgcl.get_aug_views()
            add_loss_dict = defaultdict(float)
            s = 0
            train_start = time()
            batch_num = len(train_cf) // args.batch_size
            for _ in tqdm(range(batch_num), desc=f"Epoch {epoch} CF Training"):

                batch = get_feed_dict(train_cf_with_neg, s, s + args.batch_size)
                # KGCL 內部產生 augmentation views
                batch['aug_views'] = aug_views
                batch_loss, batch_loss_dict = model(batch)
                # 清 KGCL 和 RecVAE 的梯度
                optimizer_model.zero_grad()
                ## 反向傳播
                batch_loss.backward()
                ## 更新
                # optimizer_kg.step()
                optimizer_model.step()

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
                kg_loss = model.kgcl.calc_kg_loss_transE(kg_batch_head, kg_batch_relation,
                                                         kg_batch_pos_tail, kg_batch_neg_tail)
                optimizer_kg.zero_grad()
                kg_loss.backward()
                optimizer_kg.step()
                kg_total_loss += kg_loss.item()
            kg_time = time() - kg_start

            logger.info("Epoch %04d | CF Time: %.1fs | KG Time: %.1fs | Mean KG Loss: %.4f",
                        epoch, cf_time, kg_time, kg_total_loss / n_kg_batch)

            # ----- 評估 -----
            if epoch % test_interval == 0 and epoch >= 0:

                model.kgcl.eval()
                test_start = time()
                with torch.no_grad():
                    ret = evaluator.test(model, user_dict, n_params)
                test_time = time() - test_start
                results = PrettyTable()
                results.field_names = ["Epoch", "CF Time", "KG Time", "Recall", "NDCG", "Precision", "Hit Ratio"]
                results.add_row([epoch, f"{cf_time:.1f}", f"{kg_time:.1f}",
                                 ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']])
                logger.info("\n" + str(results))
                #-------
                recall_at_20 = ret['recall'][0]
                u_emb, i_emb = model.generate()
                p_u = F.normalize(model.proj_mlp(u_emb), dim=1) * model.alpha
                mlp_scores = p_u @ i_emb.t()
                train_pairs = train_cf
                test_pairs = test_cf
                plotter.record_mlp(
                    epoch,
                    mlp_scores,
                    u_emb,
                    i_emb,
                    train_pairs,
                    test_pairs,
                    recall_at_20
                )

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