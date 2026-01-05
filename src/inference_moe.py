import torch
import torch.nn as nn
import os
from moe_kan_lib import KANNetwork 

class MedicalMoESystem(nn.Module):
    def __init__(self, model_dir='trained_models', router_path='router_weights.pth', num_experts=32, device='cuda'):
        super().__init__()
        self.device = device
        self.num_experts = num_experts
        
        # 1. FIX: Changed hidden layer from 64 to 128 to match your trained checkpoint
        self.router = nn.Sequential(
            nn.Linear(384, 128),  # Matching torch.Size([128, 384])
            nn.ReLU(),
            nn.Linear(128, num_experts) # Matching torch.Size([32, 128])
        ).to(device)
        
        self.router.load_state_dict(torch.load(router_path, map_location=device))
        self.router.eval()

        # 2. Experts: Updated architecture to match KAN_LAYERS = [384, 256, 128]
        self.experts = nn.ModuleList([
            KANNetwork([384, 256, 128], grid_size=5).to(device) 
            for _ in range(num_experts)
        ])
        
        # 3. Final Aggregator
        self.final_layers = nn.ModuleList([
            nn.Linear(128, 19756).to(device)
            for _ in range(num_experts)
        ])

        print("Loading 32 Experts...")
        for i in range(num_experts):
            path = os.path.join(model_dir, f'expert_{i}_best.pth')
            # Using weights_only=False because your save format includes custom stats dictionaries
            state = torch.load(path, map_location=device, weights_only=False)
            self.experts[i].load_state_dict(state['expert_state'])
            self.final_layers[i].load_state_dict(state['final_state'])
            self.experts[i].eval()
            self.final_layers[i].eval()

    @torch.no_grad()
    def predict(self, x):
        x = x.to(self.device)
        router_logits = self.router(x)
        expert_indices = torch.argmax(router_logits, dim=1)
        
        final_logits = torch.zeros(x.size(0), 19756, device=self.device)
        
        for i in range(self.num_experts):
            mask = (expert_indices == i)
            if mask.sum() == 0:
                continue
            
            expert_in = x[mask]
            features = self.experts[i](expert_in)
            logits = self.final_layers[i](features)
            final_logits[mask] = logits
            
        return torch.argmax(final_logits, dim=1)