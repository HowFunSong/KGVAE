import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict

# 設定 matplotlib log level，避免顯示過多警告
mpl.set_loglevel('warning')

class Plotter:
    def __init__(
        self,
        out_dir: str = "plots",
        max_users: int = 1024,
        max_items: int = 1024
    ):
        """
        Plotter 初始化：
        - out_dir    : 圖片儲存路徑
        - max_users  : heatmap/overlay 顯示的最大使用者數（取前幾筆）
        - max_items  : heatmap/overlay 顯示的最大項目數（取前幾筆）
        """
        self.best_recall = -np.inf  # 記錄目前最佳 recall
        self.epochs = []            # 紀錄所有 epoch
        self.recalls = []           # 紀錄每個 epoch 的 recall
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # heatmap/overlay 時，只取前 max_users × max_items 的小區域
        self.max_users = max_users
        self.max_items = max_items

    def record(
        self,
        epoch: int,
        ease_scores,
        user_emb,
        item_emb,
        train_cf_pairs: np.ndarray,
        test_cf_pairs: np.ndarray,
        recall_score: float
    ):
        """
        Record metrics 並且當 recall 有所提升時，輸出各種熱圖與箱型圖：

        Heatmap（皆取子矩陣以節省記憶體）：
          1. ease_scores 原始 heatmap
          2. user-item (dot product) 原始 heatmap
          3. combined (ease * ui_scores) 原始 heatmap
          4. ease_scores_train_mask：train 區塊遮蔽後的 heatmap
          5. ui_scores_train_mask：train 區塊遮蔽後的 heatmap
          6. combined_train_mask：train 區塊遮蔽後的 heatmap
          7. RGB 互動圖：train only / test only / overlay

        箱型圖（基於全域分數，不受子矩陣截斷）：
          - Ease 正負分布比較
          - UI    正負分布比較
          - Combined 正負分布比較

        參數：
          - epoch: 當前訓練輪次
          - ease_scores: Tensor or ndarray, shape [n_users, n_items]
          - user_emb:     Tensor or ndarray, shape [n_users, emb_dim]
          - item_emb:     Tensor or ndarray, shape [n_items, emb_dim]
          - train_cf_pairs, test_cf_pairs: np.ndarray of [u, i] 配對
          - recall_score: float
        """
        import torch

        # 1. 轉為 numpy array
        ease = (
            ease_scores.detach().cpu().numpy()
            if hasattr(ease_scores, 'detach')
            else np.array(ease_scores)
        )
        u_e = (
            user_emb.detach().cpu().numpy()
            if hasattr(user_emb, 'detach') else np.array(user_emb)
        )
        i_e = (
            item_emb.detach().cpu().numpy()
            if hasattr(item_emb, 'detach') else np.array(item_emb)
        )

        # 2. 全域矩陣大小
        n_users, n_items = ease.shape

        # 3. 計算截斷子矩陣範圍
        u_lim = min(n_users, self.max_users)
        i_lim = min(n_items, self.max_items)

        # 4. 取子矩陣以供 heatmap
        ease_small = ease[:u_lim, :i_lim]

        # 5. 計算 UI score（dot product）並截斷
        ui_full = np.dot(u_e, i_e.T)
        ui_small = ui_full[:u_lim, :i_lim]

        # 6. 計算 combined 分數，並截斷
        combined_small = ease_small * ui_small

        # 7. 建立子矩陣下的 RGB 互動圖容器
        train_matrix = np.zeros((u_lim, i_lim, 3), dtype=np.float32)
        test_matrix  = np.zeros((u_lim, i_lim, 3), dtype=np.float32)
        overlay_matrix = np.zeros((u_lim, i_lim, 3), dtype=np.float32)
        for u, i in train_cf_pairs:
            if u < u_lim and i < i_lim:
                train_matrix[u, i]   = [0.0, 1.0, 0.0]  # 綠色代表 train
                overlay_matrix[u, i] = [0.0, 1.0, 0.0]
        for u, i in test_cf_pairs:
            if u < u_lim and i < i_lim:
                test_matrix[u, i]    = [0.0, 1.0, 0.0]  # 洋紅代表 test
                overlay_matrix[u, i] = [0.0, 1.0, 0.0]

        # 8. 紀錄 epoch, recall
        self.epochs.append(epoch)
        self.recalls.append(recall_score)

        # 9. 若 recall 創新高，才進行繪圖與儲存
        if recall_score > self.best_recall:
            self.best_recall = recall_score

            # ------ Heatmaps ------
            self._plot_heatmap(ease_small,     'ease_scores',            epoch)
            self._plot_heatmap(ui_small,       'ui_scores',              epoch)
            self._plot_heatmap(combined_small, 'combined_scores',        epoch)

            # mask train 區塊後的 heatmap
            train_ease = ease_small.copy()
            train_ui   = ui_small.copy()
            train_comb = combined_small.copy()
            for u, i in train_cf_pairs:
                if u < u_lim and i < i_lim:
                    train_ease[u, i] = 0.0
                    train_ui[u, i]   = 0.0
                    train_comb[u, i] = 0.0
            self._plot_heatmap(train_ease, 'ease_scores_train_mask', epoch)
            self._plot_heatmap(train_ui,   'ui_scores_train_mask',   epoch)
            self._plot_heatmap(train_comb, 'combined_train_mask',    epoch)

            # RGB 互動圖
            self._plot_rgb_heatmap(train_matrix,   'train_interaction',     epoch)
            self._plot_rgb_heatmap(test_matrix,    'test_interaction',      epoch)
            self._plot_rgb_heatmap(overlay_matrix, 'train_test_overlay',    epoch)

            # ------ 全域箱型圖統計 ------
            # 建立使用者對 train/test 互動的 lookup
            train_dict = defaultdict(set)
            test_dict  = defaultdict(set)
            for u, i in train_cf_pairs:
                train_dict[int(u)].add(int(i))
            for u, i in test_cf_pairs:
                test_dict[int(u)].add(int(i))

            # 針對每個 test 正樣本，隨機抽一個負樣本 index
            neg_indices = []
            for u, _ in test_cf_pairs:
                while True:
                    j = np.random.randint(0, n_items)
                    if j not in train_dict[u] and j not in test_dict[u]:
                        neg_indices.append(j)
                        break

            # 收集 globlal 正負分數
            ease_pos = [ease[u, i] for u, i in test_cf_pairs]
            ease_neg = [ease[u, j] for (u, _), j in zip(test_cf_pairs, neg_indices)]
            ui_pos   = [u_e[u].dot(i_e[i]) for u, i in test_cf_pairs]
            ui_neg   = [u_e[u].dot(i_e[j]) for (u, _), j in zip(test_cf_pairs, neg_indices)]
            comb_pos = [ease[u, i] * ui_pos[idx] for idx, (u, i) in enumerate(test_cf_pairs)]
            comb_neg = [ease[u, j] * ui_neg[idx] for idx, ((u, _), j) in enumerate(zip(test_cf_pairs, neg_indices))]

            # 繪製 3 欄子圖
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].boxplot([ease_pos, ease_neg], labels=['Pos', 'Neg'])
            axes[0].set_title('Ease Scores')
            axes[1].boxplot([ui_pos, ui_neg], labels=['Pos', 'Neg'])
            axes[1].set_title('UI Embedding')
            axes[2].boxplot([comb_pos, comb_neg], labels=['Pos', 'Neg'])
            axes[2].set_title('Combined')
            fig.suptitle(f'Pos vs Neg (epoch={epoch})')
            fn = os.path.join(self.out_dir, f'pos_neg_compare.png')
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.savefig(fn)
            plt.close(fig)

            print(f"Epoch {epoch}: new best recall {recall_score:.4f}")

    def record_mlp(
        self,
        epoch: int,
        mlp_scores,
        user_emb,
        item_emb,
        train_cf_pairs: np.ndarray,
        test_cf_pairs: np.ndarray,
        recall_score: float
    ):
        """
        Record metrics 並且當 recall 有所提升時，輸出各種熱圖與箱型圖：

        Heatmap（皆取子矩陣以節省記憶體）：
          1. ease_scores 原始 heatmap
          2. user-item (dot product) 原始 heatmap
          3. combined (ease * ui_scores) 原始 heatmap
          4. ease_scores_train_mask：train 區塊遮蔽後的 heatmap
          5. ui_scores_train_mask：train 區塊遮蔽後的 heatmap
          6. combined_train_mask：train 區塊遮蔽後的 heatmap
          7. RGB 互動圖：train only / test only / overlay

        箱型圖（基於全域分數，不受子矩陣截斷）：
          - Ease 正負分布比較
          - UI    正負分布比較
          - Combined 正負分布比較

        參數：
          - epoch: 當前訓練輪次
          - ease_scores: Tensor or ndarray, shape [n_users, n_items]
          - user_emb:     Tensor or ndarray, shape [n_users, emb_dim]
          - item_emb:     Tensor or ndarray, shape [n_items, emb_dim]
          - train_cf_pairs, test_cf_pairs: np.ndarray of [u, i] 配對
          - recall_score: float
        """
        import torch

        # 1. 轉為 numpy array
        mlp = (
            mlp_scores.detach().cpu().numpy()
            if hasattr(mlp_scores, 'detach')
            else np.array(mlp_scores)
        )
        u_e = (
            user_emb.detach().cpu().numpy()
            if hasattr(user_emb, 'detach') else np.array(user_emb)
        )
        i_e = (
            item_emb.detach().cpu().numpy()
            if hasattr(item_emb, 'detach') else np.array(item_emb)
        )

        # 2. 全域矩陣大小
        n_users, n_items = mlp.shape

        # 3. 計算截斷子矩陣範圍
        u_lim = min(n_users, self.max_users)
        i_lim = min(n_items, self.max_items)

        # 4. 取子矩陣以供 heatmap
        mlp_small = mlp[:u_lim, :i_lim]

        # 5. 計算 UI score（dot product）並截斷
        ui_full = np.dot(u_e, i_e.T)
        ui_small = ui_full[:u_lim, :i_lim]

        # 6. 計算 combined 分數，並截斷
        combined_small = mlp_small + ui_small

        # 7. 建立子矩陣下的 RGB 互動圖容器
        train_matrix = np.zeros((u_lim, i_lim, 3), dtype=np.float32)
        test_matrix = np.zeros((u_lim, i_lim, 3), dtype=np.float32)
        overlay_matrix = np.zeros((u_lim, i_lim, 3), dtype=np.float32)
        for u, i in train_cf_pairs:
            if u < u_lim and i < i_lim:
                train_matrix[u, i]   = [0.0, 1.0, 0.0]  # 綠色代表 train
                overlay_matrix[u, i] = [0.0, 1.0, 0.0]
        for u, i in test_cf_pairs:
            if u < u_lim and i < i_lim:
                test_matrix[u, i]    = [0.0, 1.0, 0.0]  # 洋紅代表 test
                overlay_matrix[u, i] = [0.0, 1.0, 0.0]

        # 8. 紀錄 epoch, recall
        self.epochs.append(epoch)
        self.recalls.append(recall_score)

        # 9. 若 recall 創新高，才進行繪圖與儲存
        if recall_score > self.best_recall:
            self.best_recall = recall_score

            # ------ Heatmaps ------
            self._plot_heatmap(mlp_small,     'mlp_scores',            epoch)
            self._plot_heatmap(ui_small,       'ui_scores',              epoch)
            self._plot_heatmap(combined_small, 'combined_scores',        epoch)

            # mask train 區塊後的 heatmap
            train_ease = mlp_small.copy()
            train_ui   = ui_small.copy()
            train_comb = combined_small.copy()
            for u, i in train_cf_pairs:
                if u < u_lim and i < i_lim:
                    train_ease[u, i] = 0.0
                    train_ui[u, i]   = 0.0
                    train_comb[u, i] = 0.0
            self._plot_heatmap(train_ease, 'mlp_scores_train_mask', epoch)
            self._plot_heatmap(train_ui,   'ui_scores_train_mask',   epoch)
            self._plot_heatmap(train_comb, 'combined_train_mask',    epoch)

            # RGB 互動圖
            self._plot_rgb_heatmap(train_matrix,   'train_interaction',     epoch)
            self._plot_rgb_heatmap(test_matrix,    'test_interaction',      epoch)
            self._plot_rgb_heatmap(overlay_matrix, 'train_test_overlay',    epoch)

            # ------ 全域箱型圖統計 ------
            # 建立使用者對 train/test 互動的 lookup
            train_dict = defaultdict(set)
            test_dict  = defaultdict(set)
            for u, i in train_cf_pairs:
                train_dict[int(u)].add(int(i))
            for u, i in test_cf_pairs:
                test_dict[int(u)].add(int(i))

            # 針對每個 test 正樣本，隨機抽一個負樣本 index
            neg_indices = []
            for u, _ in test_cf_pairs:
                while True:
                    j = np.random.randint(0, n_items)
                    if j not in train_dict[u] and j not in test_dict[u]:
                        neg_indices.append(j)
                        break

            # 收集 globlal 正負分數
            mlp_pos = [mlp[u, i] for u, i in test_cf_pairs]
            mlp_neg = [mlp[u, j] for (u, _), j in zip(test_cf_pairs, neg_indices)]
            ui_pos = [u_e[u].dot(i_e[i]) for u, i in test_cf_pairs]
            ui_neg = [u_e[u].dot(i_e[j]) for (u, _), j in zip(test_cf_pairs, neg_indices)]
            comb_pos = [mlp[u, i] * ui_pos[idx] for idx, (u, i) in enumerate(test_cf_pairs)]
            comb_neg = [mlp[u, j] * ui_neg[idx] for idx, ((u, _), j) in enumerate(zip(test_cf_pairs, neg_indices))]

            # 繪製 3 欄子圖
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].boxplot([mlp_pos, mlp_neg], labels=['Pos', 'Neg'])
            axes[0].set_title('alignment Scores')
            axes[1].boxplot([ui_pos, ui_neg], labels=['Pos', 'Neg'])
            axes[1].set_title('UI Embedding')
            axes[2].boxplot([comb_pos, comb_neg], labels=['Pos', 'Neg'])
            axes[2].set_title('Combined')
            fig.suptitle(f'Pos vs Neg (epoch={epoch})')
            fn = os.path.join(self.out_dir, f'pos_neg_compare.png')
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.savefig(fn)
            plt.close(fig)

            print(f"Epoch {epoch}: new best recall {recall_score:.4f}")

    def _plot_heatmap(self, matrix: np.ndarray, title: str, epoch: int, cmap: str = 'viridis'):
        """
        繪製 2D heatmap 並儲存
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, aspect='auto', cmap=cmap)
        plt.colorbar()
        plt.title(f"{title} (epoch {epoch})")
        plt.xlabel('Items')
        plt.ylabel('Users')
        fname = os.path.join(self.out_dir, f"{title}.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def _plot_rgb_heatmap(self, rgb_matrix: np.ndarray, title: str, epoch: int):
        """
        繪製 RGB 三通道互動圖並儲存
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(np.clip(rgb_matrix, 0, 1), aspect='auto')
        plt.title(f"{title} (epoch {epoch})")
        plt.xlabel('Items')
        plt.ylabel('Users')
        import matplotlib.patches as mpatches
        handles = []
        lower = title.lower()
        if 'train_interaction' in lower:
            handles.append(mpatches.Patch(color='green', label='Train'))
        if 'test_interaction' in lower:
            handles.append(mpatches.Patch(color='magenta', label='Test'))
        if 'train_test_overlay' in lower:
            handles.append(mpatches.Patch(color='yellow', label='Overlay'))
        if handles:
            plt.legend(handles=handles, loc='upper right', facecolor='white')
        fname = os.path.join(self.out_dir, f"{title}.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
