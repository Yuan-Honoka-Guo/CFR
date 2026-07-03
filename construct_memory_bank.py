import argparse
import os
import torch
import numpy as np
from tqdm import tqdm, trange
from sklearn.cluster import MiniBatchKMeans  # Used for the clustering method
from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from models.sampler import ApproximateGreedyCoresetSampler

def set_seeds(sid=115):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)

class MemoryBank:
    def __init__(self, reduction_method='patchcore', device='cpu'):
        """
        Args:
            reduction_method (str): 'patchcore', 'random' or 'kmeans'.
            device (str): Device to store the final bank on.
        """
        self.reduction_method = reduction_method
        self.device = device
        self.features_buffer = []  # Temporary list to store features during training
        self.memory_bank = None    # The final tensor (N, D)
        self.sampler = ApproximateGreedyCoresetSampler(percentage=0.1, device=device)  # For patchcore method

    def add_features(self, features):
        """
        Accumulates features during the training loop.
        Args:
            features (torch.Tensor): Shape (B, N_patches, D) or (N, D).
        """
        # Flatten to (Total_Patches, D) and move to CPU to save GPU memory during accumulation
        flat_features = features.reshape(-1, features.shape[-1]).detach().cpu()
        self.features_buffer.append(flat_features)

    def fit(self):
        """
        Process the accumulated features to build the final Memory Bank.
        """
        if not self.features_buffer:
            print("No features collected.")
            return

        # Concatenate all collected features
        all_features = torch.cat(self.features_buffer, dim=0)
        total_samples = all_features.shape[0]
        
        print(f"Total extracted features: {total_samples}")
        self.target_size = total_samples * 0.01  # For example, keep 1% of features
        print(f"Reducing features using method: {self.reduction_method}...")

        if self.reduction_method == 'patchcore':
            # PatchCore coreset sampling (greedy k-center) on a random projection to speed up distance calcs.
            self.memory_bank = self._patchcore_downsample(all_features).to(self.device)

        elif self.reduction_method == 'random':
            # Random Sampling
            indices = torch.randperm(total_samples)[:self.target_size]
            self.memory_bank = all_features[indices].to(self.device)

        elif self.reduction_method == 'kmeans':
            # K-Means Clustering (using sklearn for efficiency on CPU)
            # We use the cluster centers as the memory bank items
            kmeans = MiniBatchKMeans(n_clusters=self.target_size, batch_size=4096, n_init='auto')
            kmeans.fit(all_features.numpy())
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            self.memory_bank = centers.to(self.device)
        
        else:
            raise ValueError(f"Unknown method: {self.reduction_method}")
            
        # Clear buffer to free memory
        self.features_buffer = []
        print(f"Memory Bank built. Final shape: {self.memory_bank.shape}")

    def _patchcore_downsample(self, all_features, proj_dim=128):
        """
        Greedy coreset sampling as used in PatchCore.
        Projects features to a lower dimension before farthest-first traversal to avoid OOM.
        """
        return self.sampler.run(all_features)

    def get_nearest_neighbors(self, query_features, k=1):
        """
        Matches query features to the Memory Bank.
        
        Args:
            query_features (torch.Tensor): Shape (N_query, D).
            k (int): Number of nearest neighbors to retrieve.
            
        Returns:
            dist (torch.Tensor): Distance to the top-k neighbors. Shape (N_query, k).
            features (torch.Tensor): The actual top-k feature vectors. Shape (N_query, k, D).
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory Bank is empty. Call fit() first.")
        
        # Ensure query is on the same device
        query_features = query_features.to(self.device)
        
        # Calculate Euclidean distance matrix: (N_query, N_bank)
        # Using cdist is memory intensive. For large banks, iterate or use Faiss.
        # Here we use a chunked approach if necessary, but standard cdist for simplicity:
        distances = torch.cdist(query_features, self.memory_bank, p=2) # L2 Distance

        # Get Top-K smallest distances
        # values: (N_query, k), indices: (N_query, k)
        topk_dist, topk_indices = torch.topk(distances, k=k, dim=1, largest=False)

        # Retrieve the actual features
        # Expand indices to match feature dimension D
        # indices shape: (N_query, k) -> (N_query, k, D)
        indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, self.memory_bank.shape[-1])
        
        # Gather features from memory bank
        # We need to expand memory_bank to gather: (1, N_bank, D) -> (N_query, N_bank, D) is too big.
        # Instead, simple indexing:
        topk_features = self.memory_bank[topk_indices] # PyTorch handles (N_query, k) indexing elegantly

        return topk_dist, topk_features

class KeyValueMemoryBank:
    def __init__(self, reduction_method='patchcore', device='cpu'):
        """
        Stores paired key/value features for attention-based retrieval.
        Keys are expected to be 3D features and values 2D features.
        """
        self.reduction_method = reduction_method
        self.device = device
        self.keys_buffer = []
        self.values_buffer = []
        self.memory_bank = None
        self.sampler = ApproximateGreedyCoresetSampler(percentage=0.1, device=device)

    def add_features(self, keys, values):
        flat_keys = keys.reshape(-1, keys.shape[-1]).detach().cpu()
        flat_values = values.reshape(-1, values.shape[-1]).detach().cpu()
        self.keys_buffer.append(flat_keys)
        self.values_buffer.append(flat_values)

    def fit(self):
        if not self.keys_buffer:
            print("No key/value features collected.")
            return

        all_keys = torch.cat(self.keys_buffer, dim=0)
        all_values = torch.cat(self.values_buffer, dim=0)
        total_samples = all_keys.shape[0]

        print(f"Total extracted key/value pairs: {total_samples}")
        target_size = int(total_samples * 0.01)
        print(f"Reducing pairs using method: {self.reduction_method}...")

        if self.reduction_method == 'patchcore':
            sampled_keys, indices = self.sampler.run_with_indices(all_keys)
            indices_t = torch.as_tensor(indices, dtype=torch.long)
            sampled_values = all_values[indices_t]
        elif self.reduction_method == 'random':
            indices_t = torch.randperm(total_samples)[:target_size]
            sampled_keys = all_keys[indices_t]
            sampled_values = all_values[indices_t]
        elif self.reduction_method == 'kmeans':
            raise ValueError("kmeans reduction is not supported for key/value banks.")
        else:
            raise ValueError(f"Unknown method: {self.reduction_method}")

        self.memory_bank = {
            "keys": sampled_keys.to(self.device),
            "values": sampled_values.to(self.device),
        }

        self.keys_buffer = []
        self.values_buffer = []
        print(f"Key/Value Memory Bank built. Keys shape: {self.memory_bank['keys'].shape}, Values shape: {self.memory_bank['values'].shape}")

def make_memory_bank(args):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Save setup
    shot_tag = f'_{args.few_shot}shot' if args.few_shot is not None else '_full'
    save_dir = os.path.join(args.checkpoint_savepath, args.class_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if args.bank_type == "kv":
        save_file = os.path.join(save_dir, f'memory_bank_kv{shot_tag}.pt')
    else:
        save_file = os.path.join(save_dir, f'memory_bank{shot_tag}.pt')


    # Dataloader
    train_loader = get_data_loader("train", class_name=args.class_name, img_size=224, 
                                   dataset_path=args.dataset_path, batch_size=1, 
                                   shuffle=True, few_shot=args.few_shot, few_shot_seed=args.few_shot_seed)
    
    # Feature extractors
    feature_extractor = MultimodalFeatures().to(device)
    feature_extractor.eval() # Ensure eval mode

    # Initialize Memory Bank
    # reduction_method: 'patchcore', 'random', or 'kmeans'
    if args.bank_type == "kv":
        memory_bank = KeyValueMemoryBank(reduction_method=args.sampling_method, device=device)
    else:
        memory_bank = MemoryBank(reduction_method=args.sampling_method, device=device)

    # We might need to downsample online if images are huge (224x224 patches)
    # Keeping 10% of patches randomly during the loop prevents RAM explosion before final fitting
    if args.few_shot:
        print(f"--- Few-Shot Mode Activated ({args.few_shot} samples) ---")
        print("Disabling online subsampling to preserve all information from few samples.")
        online_subsample_ratio = 1.0 # Keep 100% of patches
    else:
        online_subsample_ratio = 0.1 # Randomly drop patches to save RAM during full training

    with torch.no_grad():
        # Note: Usually 1 epoch is enough to build a bank from training data
        for (rgb, pc, _), _ in tqdm(train_loader, desc=f'Extracting features ({args.class_name})'):
            rgb, pc = rgb.to(device), pc.to(device)

            # Extract features
            # Output should be (B, N, D). 
            rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb, pc)

            # --- Online Subsampling (Optional but recommended for high-res) ---
            # Flatten: (B * H * W, D)
            flat_rgb = rgb_patch.reshape(-1, rgb_patch.shape[-1])
            flat_xyz = xyz_patch.reshape(-1, xyz_patch.shape[-1])

            if online_subsample_ratio < 1.0:
                num_keep = int(flat_rgb.shape[0] * online_subsample_ratio)
                if num_keep > 0:
                    idx = torch.randperm(flat_rgb.shape[0])[:num_keep]
                    flat_rgb = flat_rgb[idx]
                    flat_xyz = flat_xyz[idx]

            # Add to Memory Bank buffer
            if args.bank_type == "kv":
                memory_bank.add_features(flat_xyz, flat_rgb)
            else:
                memory_bank.add_features(flat_rgb)

    # Finalize the Memory Bank (Clustering/Sampling happens here)
    print("Finalizing Memory Bank...")
    memory_bank.fit()

    # Save to disk
    torch.save(memory_bank.memory_bank, save_file)
    print(f"Saved Memory Bank to {save_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Making Memory Bank on a dataset.')

    parser.add_argument('--dataset_path', default='/path/to/your/dataset/mvtec3d', type=str)
    parser.add_argument('--checkpoint_savepath', default='/path/to/your/checkpoints/memory_bank', type=str)
    parser.add_argument('--class_name', default='bagel', type=str)
    
    # New arguments for Memory Bank
    parser.add_argument('--sampling_method', default='patchcore', choices=['patchcore', 'random', 'kmeans'], 
                        help='Method to reduce feature count: PatchCore coreset, random sampling, or k-means clustering')
    
    parser.add_argument('--bank_type', default='kv', choices=['rgb', 'kv'],
                        help='rgb: store 2D features only; kv: store 3D keys + 2D values for attention retrieval')

    # Few shot args (kept from your snippet)
    parser.add_argument('--few_shot', default=None, type=int)
    parser.add_argument('--few_shot_seed', default=42, type=int)

    args = parser.parse_args()
    make_memory_bank(args)
