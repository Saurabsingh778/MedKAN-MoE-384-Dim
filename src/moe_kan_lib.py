import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EfficientKANLinear(nn.Module):
    """
    An efficient Linear KAN layer (B-Splines).
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Learnable parameters
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features * (grid_size + spline_order)))
        
        # Scaling
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # Grid (non-learnable buffer)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.reset_parameters()

    def reset_parameters(self):
        # 1. Initialize Base Linear Layer (Kaiming)
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        
        # 2. Initialize Spline Weights (Noise)
        with torch.no_grad():
            # We need (grid_size + spline_order) coefficients per input feature
            num_coeffs = self.grid_size + self.spline_order
            
            # Generate noise: (Num_Coeffs, In, Out)
            noise = (torch.rand(num_coeffs, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
            
            # Permute to (Out, In, Num_Coeffs) and flatten to (Out, In * Num_Coeffs)
            target_shape = (self.out_features, self.in_features * num_coeffs)
            
            self.spline_weight.data.copy_(
                (self.scale_spline * noise).permute(2, 1, 0).reshape(target_shape)
            )

            
    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1] + \
                    (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]

        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def forward(self, x):
        # 1. Base activation (silu) + Linear
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # 2. B-Spline activation
        bs = self.b_splines(x) # (Batch, In, Grid+Order)
        bs = bs.view(bs.size(0), -1) # Flatten to (Batch, In * Coeffs)
        spline_output = F.linear(bs, self.spline_weight)
        
        return base_output + spline_output

class KANNetwork(nn.Module):
    """
    A single KAN Network (The Expert).
    Can be configured with different hidden sizes.
    """
    def __init__(self, layers_hidden, grid_size=5, spline_order=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                EfficientKANLinear(
                    layers_hidden[i], 
                    layers_hidden[i+1],
                    grid_size=grid_size, 
                    spline_order=spline_order
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MoEKAN(nn.Module):
    def __init__(self, input_dim, num_experts, expert_hidden_layers, output_dim, top_k=2, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.top_k = top_k
        self.device = device
        
        # 1. THE ROUTER (Gating Network)
        # Predicts weights for each expert
        self.router = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_experts)
        ).to(device)

        # 2. THE EXPERTS
        # Every expert predicts the FULL output_dim to maintain the benchmark
        self.experts = nn.ModuleList([
            KANNetwork(expert_hidden_layers, grid_size=3).to(device) 
            for _ in range(num_experts)
        ])
        
        # 3. GLOBAL PROJECTION
        # Maps the expert hidden output to the final 19,756 classes
        last_hidden = expert_hidden_layers[-1]
        self.output_projection = nn.Linear(last_hidden, output_dim).to(device)

    def forward(self, x, train=True):
        # Step 1: Routing
        logits = self.router(x)
        
        # Add noise during training to encourage exploration (Standard MoE practice)
        if train:
            noise = torch.randn_like(logits) * (1.0 / self.num_experts)
            logits = logits + noise
            
        probs = F.softmax(logits, dim=1)
        
        # Step 2: Select Top-K Experts
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=1)
        
        # Normalize top-k probabilities so they sum to 1
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Step 3: Compute Expert Outputs
        # We process the batch through selected experts
        final_output = torch.zeros(x.size(0), self.output_dim, device=self.device)
        
        for i in range(self.num_experts):
            # Find which samples in the batch were assigned to expert 'i'
            mask = (topk_indices == i).any(dim=1)
            if not mask.any():
                continue
                
            # Get the probability weight assigned to this expert for these samples
            # This is key for the gradient flow back to the router
            expert_mask = (topk_indices == i)
            # Find the specific column index (0 or 1 in Top-2)
            batch_indices, k_indices = torch.where(expert_mask)
            weights = topk_probs[batch_indices, k_indices].unsqueeze(-1)
            
            # Forward pass through Expert + Projection
            expert_hidden = self.experts[i](x[mask])
            out = self.output_projection(expert_hidden)
            
            # Accumulate weighted output
            final_output[mask] += weights * out
            
        return final_output, probs # Return probs for load balancing loss