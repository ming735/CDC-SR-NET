import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class OptimizedDualAttentionModule(nn.Module):
    def __init__(self, dim=3, dim_head=32, heads=8, img_size=224, patch_size=16):
        super(OptimizedDualAttentionModule, self).__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.img_size = img_size
        self.patch_size = patch_size
        

        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = dim * patch_size * patch_size
        

        self.pos_encoding = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        )

        self.patch_embed = nn.Conv2d(dim, self.patch_dim, kernel_size=patch_size, stride=patch_size)

        self.to_qkv_patch = nn.Linear(self.patch_dim, dim_head * heads * 3, bias=False)

        self.to_qkv_sequence = nn.Linear(dim, dim_head * heads * 3, bias=False)

        self.proj_patch = nn.Linear(dim_head * heads, self.patch_dim)
        self.proj_sequence = nn.Linear(dim_head * heads, dim)

        self.patch_reconstruct = nn.ConvTranspose2d(
            self.patch_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_decoding = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        )

        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):

        b, c, h, w = x.shape
        

        pos_encoded = self.pos_encoding(x)
        patches = self.patch_embed(pos_encoded)
        patches = rearrange(patches, 'b c h w -> b (h w) c')

        patches_norm = self.norm1(patches)

        qkv_patch = self.to_qkv_patch(patches_norm).chunk(3, dim=-1)
        q_patch, k_patch, v_patch = map(lambda t: rearrange(t, 'b n (heads d_head) -> b heads n d_head', 
                                                          heads=self.heads), qkv_patch)

        attn_weights_patch = torch.matmul(q_patch, k_patch.transpose(-2, -1)) / (self.dim_head ** 0.5)
        attn_weights_patch = F.softmax(attn_weights_patch, dim=-1)

        output_patch = torch.matmul(attn_weights_patch, v_patch)
        output_patch = rearrange(output_patch, 'b heads n d_head -> b n (heads d_head)')
        output_patch = self.proj_patch(output_patch)

        output_patch = rearrange(output_patch, 'b (h w) c -> b c h w', 
                               h=self.num_patches_h, w=self.num_patches_w)
        output_patch = self.patch_reconstruct(output_patch)

        sequence_input = rearrange(pos_encoded, 'b c h w -> b (h w) c')

        sequence_norm = self.norm2(sequence_input)

        qkv_sequence = self.to_qkv_sequence(sequence_norm).chunk(3, dim=-1)
        q_sequence, k_sequence, v_sequence = map(lambda t: rearrange(t, 'b n (heads d_head) -> b heads n d_head', 
                                                                   heads=self.heads), qkv_sequence)

        group_size = 500
        num_groups = (h * w + group_size - 1) // group_size
        
        output_sequence_parts = []
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, h * w)
            
            q_group = q_sequence[:, :, start_idx:end_idx, :]
            

            with torch.cuda.amp.autocast(enabled=False):

                attn_weights_sequence = torch.matmul(
                    q_group.float(),
                    k_sequence.float().transpose(-2, -1)
                ) / (self.dim_head ** 0.5)
                
                attn_weights_sequence = F.softmax(attn_weights_sequence, dim=-1)

                output_group = torch.matmul(
                    attn_weights_sequence.float(),
                    v_sequence.float()
                )
            
            output_sequence_parts.append(output_group)

            del q_group, attn_weights_sequence
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        output_sequence = torch.cat(output_sequence_parts, dim=2)
        output_sequence = rearrange(output_sequence, 'b heads n d_head -> b n (heads d_head)')
        output_sequence = self.proj_sequence(output_sequence)
        output_sequence = rearrange(output_sequence, 'b (h w) c -> b c h w', h=h, w=w)

        combined_output = output_patch + output_sequence

        final_attention_map = self.pos_decoding(combined_output)
        
        return final_attention_map


class MemoryManagedDualAttentionModule(nn.Module):

    def __init__(self, dim=3, dim_head=32, heads=8, img_size=224, patch_size=16):
        super(MemoryManagedDualAttentionModule, self).__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.img_size = img_size
        self.patch_size = patch_size

        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = dim * patch_size * patch_size

        self.pos_encoding = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        )

        self.patch_embed = nn.Conv2d(dim, self.patch_dim, kernel_size=patch_size, stride=patch_size)

        self.to_qkv_patch = nn.Linear(self.patch_dim, dim_head * heads * 3, bias=False)

        self.to_qkv_sequence = nn.Linear(dim, dim_head * heads * 3, bias=False)

        self.proj_patch = nn.Linear(dim_head * heads, self.patch_dim)
        self.proj_sequence = nn.Linear(dim_head * heads, dim)

        self.patch_reconstruct = nn.ConvTranspose2d(
            self.patch_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_decoding = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        )

        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(dim)

    def compute_patch_attention(self, patches_norm):

        qkv_patch = self.to_qkv_patch(patches_norm).chunk(3, dim=-1)
        q_patch, k_patch, v_patch = map(lambda t: rearrange(t, 'b n (heads d_head) -> b heads n d_head', 
                                                          heads=self.heads), qkv_patch)

        attn_weights_patch = torch.matmul(q_patch, k_patch.transpose(-2, -1)) / (self.dim_head ** 0.5)
        attn_weights_patch = F.softmax(attn_weights_patch, dim=-1)
        

        output_patch = torch.matmul(attn_weights_patch, v_patch)
        output_patch = rearrange(output_patch, 'b heads n d_head -> b n (heads d_head)')
        output_patch = self.proj_patch(output_patch)

        return output_patch

    def compute_sequence_attention(self, sequence_norm, h, w):

        qkv_sequence = self.to_qkv_sequence(sequence_norm).chunk(3, dim=-1)
        q_sequence, k_sequence, v_sequence = map(lambda t: rearrange(t, 'b n (heads d_head) -> b heads n d_head', 
                                                                   heads=self.heads), qkv_sequence)

        group_size = 50
        num_groups = (h * w + group_size - 1) // group_size
        
        output_sequence_parts = []
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, h * w)
            
            q_group = q_sequence[:, :, start_idx:end_idx, :]

            output_group = torch.utils.checkpoint.checkpoint(
                self._compute_attention_group,
                q_group, k_sequence, v_sequence,
                preserve_rng_state=False
            )
            
            output_sequence_parts.append(output_group)
        

        output_sequence = torch.cat(output_sequence_parts, dim=2)
        output_sequence = rearrange(output_sequence, 'b heads n d_head -> b n (heads d_head)')
        output_sequence = self.proj_sequence(output_sequence)
        
        return output_sequence

    def _compute_attention_group(self, q_group, k_sequence, v_sequence):

        attn_weights_sequence = torch.matmul(q_group, k_sequence.transpose(-2, -1)) / (self.dim_head ** 0.5)
        attn_weights_sequence = F.softmax(attn_weights_sequence, dim=-1)
        

        output_group = torch.matmul(attn_weights_sequence, v_sequence)
        
        return output_group

    def forward(self, x):

        b, c, h, w = x.shape

        pos_encoded = self.pos_encoding(x)

        patches = self.patch_embed(pos_encoded)
        patches = rearrange(patches, 'b c h w -> b (h w) c')

        patches_norm = self.norm1(patches)

        output_patch = self.compute_patch_attention(patches_norm)
        

        output_patch = rearrange(output_patch, 'b (h w) c -> b c h w', 
                               h=self.num_patches_h, w=self.num_patches_w)
        output_patch = self.patch_reconstruct(output_patch)

        sequence_input = rearrange(pos_encoded, 'b c h w -> b (h w) c')

        sequence_norm = self.norm2(sequence_input)

        output_sequence = self.compute_sequence_attention(sequence_norm, h, w)
        output_sequence = rearrange(output_sequence, 'b (h w) c -> b c h w', h=h, w=w)

        combined_output = output_patch + output_sequence

        final_attention_map = self.pos_decoding(combined_output)
        
        return final_attention_map


class GSC(nn.Module):

    def __init__(self, dim=3, dim_head=32, heads=8, img_size=224, patch_size=16, mlp_ratio=4):
        super(GSC, self).__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.img_size = img_size

        self.dual_attention = MemoryManagedDualAttentionModule(
            dim, dim_head, heads, img_size, patch_size
        )

        ff_dim = dim * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, ff_dim, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(ff_dim, ff_dim, 3, 1, 1, bias=False, groups=ff_dim),
            nn.GELU(),
            nn.Conv2d(ff_dim, dim, 1, 1, bias=False),
        )
        
        # 层归一化
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual1 = x

        attention_output = self.dual_attention(x)

        output = self.norm(attention_output + residual1)

        residual2 = output
        output = self.ffn(output)
        output = output + residual2
        
        return output
