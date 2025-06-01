import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.set_loglevel('warning')

class Plotter:
    def __init__(self, out_dir="plots"):
        self.best_recall = -np.inf
        self.epochs = []
        self.recalls = []
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def record(self, epoch, ease_scores, user_emb, item_emb,
               train_cf_pairs, test_cf_pairs, recall_score):
        """
        Record metrics and, if recall improves, output the requested heatmaps:
        1. ease_scores heatmap
        2. user-item score heatmap (dot product of embeddings)
        3. element-wise product of ease_scores and user-item scores
        4. separate interaction matrices (train only, test only, train+test overlay)

        ease_scores: torch.Tensor or ndarray [n_users, n_items]
        user_emb:    torch.Tensor [n_users, emb_dim]
        item_emb:    torch.Tensor [n_items, emb_dim]
        train_cf_pairs, test_cf_pairs: numpy arrays [[u, i], ...]
        recall_score: float
        """
        import torch

        # Convert tensors to numpy arrays
        ease = (ease_scores.detach().cpu().numpy()
                if hasattr(ease_scores, 'detach')
                else np.array(ease_scores))
        u_e = (user_emb.detach().cpu().numpy()
               if hasattr(user_emb, 'detach')
               else np.array(user_emb))
        i_e = (item_emb.detach().cpu().numpy()
               if hasattr(item_emb, 'detach')
               else np.array(item_emb))

        n_users, n_items = ease.shape

        # Compute user-item score matrix via dot product
        ui_scores = np.dot(u_e, i_e.T)  # shape [n_users, n_items]

        # Combined scores: ease * ui_scores
        combined = ease * ui_scores

        # Build separate interaction matrices on black background
        train_matrix = np.zeros((n_users, n_items, 3), dtype=np.float32)
        test_matrix = np.zeros((n_users, n_items, 3), dtype=np.float32)
        overlay_matrix = np.zeros((n_users, n_items, 3), dtype=np.float32)

        # Train only: green
        for u, i in train_cf_pairs:
            train_matrix[u, i] = np.array([0.0, 1.0, 0.0])
            overlay_matrix[u, i] = np.array([0.0, 1.0, 0.0])

        # Test only: magenta (red + blue)
        for u, i in test_cf_pairs:
            test_matrix[u, i] = np.array([1.0, 0.0, 1.0])
            overlay_matrix[u, i] = np.array([1.0, 0.0, 1.0])

        # Save epoch and recall
        self.epochs.append(epoch)
        self.recalls.append(recall_score)

        # If recall improves, plot and save
        if recall_score > self.best_recall:
            self.best_recall = recall_score
            # 1. Ease scores heatmap
            self._plot_heatmap(ease, 'Ease Scores', epoch, cmap='viridis')
            # 2. User-item scores heatmap
            self._plot_heatmap(ui_scores, 'User-Item Scores', epoch, cmap='viridis')
            # 3. Combined heatmap
            self._plot_heatmap(combined, 'Combined Ease-UI Scores', epoch, cmap='viridis')
            # 4. Evaluate ease score insufficiency by masking train interactions in ease matrix
            # ==== 新增：Evaluate ease score insufficiency by zeroing train interactions in ease matrix ====
            # Make a copy of ease and set train positions to zero
            train_ease = ease.copy()
            for u, i in train_cf_pairs:
                train_ease[u, i] = 0.0
            self._plot_heatmap(train_ease, 'Ease_Scores_Train_Mask', epoch, cmap='viridis')

            # 4a. Train only interaction Train only interaction
            self._plot_rgb_heatmap(train_matrix, 'Train Interaction', epoch)
            # 4b. Test only interaction
            self._plot_rgb_heatmap(test_matrix, 'Test Interaction', epoch)
            # 4c. Train+Test overlay
            self._plot_rgb_heatmap(overlay_matrix, 'Train_Test_Overlay', epoch)
            print(f"Epoch {epoch}: new best recall {recall_score:.4f}")

    def _plot_heatmap(self, matrix, title, epoch, cmap='viridis'):
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, aspect='auto', cmap=cmap)
        plt.colorbar()
        plt.title(f"{title} (epoch {epoch})")
        plt.xlabel('Items')
        plt.ylabel('Users')
        fname = os.path.join(self.out_dir, f"{title.replace(' ', '_').lower()}_e{epoch}.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def _plot_rgb_heatmap(self, rgb_matrix, title, epoch):
        plt.figure(figsize=(8, 6))
        img = np.clip(rgb_matrix, 0, 1)
        plt.imshow(img, aspect='auto')
        plt.title(f"{title} (epoch {epoch})")
        plt.xlabel('Items')
        plt.ylabel('Users')

        # Create legend proxies
        import matplotlib.patches as mpatches
        handles = []
        lower = title.lower()
        if 'train' in lower and 'overlay' not in lower:
            handles.append(mpatches.Patch(color='green', label='Train Interaction'))
        if 'test' in lower and 'overlay' not in lower:
            handles.append(mpatches.Patch(color='magenta', label='Test Interaction'))
        if 'overlay' in lower:
            handles.append(mpatches.Patch(color='green', label='Train'))
            handles.append(mpatches.Patch(color='magenta', label='Test'))
        if handles:
            plt.legend(handles=handles, loc='upper right', facecolor='white')

        fname = os.path.join(self.out_dir, f"{title.replace(' ', '_').lower()}_e{epoch}.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
