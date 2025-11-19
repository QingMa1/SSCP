import typing as t
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from mmengine.model import BaseModule

# MSA_processed_x + SRGA_processed(MSA_x) -> CSA
class SSCP(BaseModule):
    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
            in_channel=256,
            in_spatial=256, 
            cha_ratio=8, 
            spa_ratio=8, 
            down_ratio=8

    ):
        super(SMSA_RGAs_add22_PCSA, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                # dimensionality reduction
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)


        self.in_channel = in_channel
        self.in_spatial = in_spatial

        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = in_spatial // spa_ratio
        
        # Embedding functions for original features
        
        self.gx_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                    kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )
        
        
        # Embedding functions for relation features
    
        self.gg_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
                    kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU()
        )
        
        
        # Networks for learning attention weights
        
        num_channel_s = 1 + self.inter_spatial
        self.W_spatial = nn.Sequential(
            nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
                    kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel_s//down_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
                    kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1)
            )

        # Embedding functions for modeling relations
        
        self.theta_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                            kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )
        self.phi_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                        kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The dim of x is (B, C, H, W)
        """

        # MSA_processed_x + SRGA_processed(MSA_x) -> CSA

        #MSA
        b, c, h_, w_ = x.size()
        # Spatial attention priority calculation:SMSA
        # (B, C, H)
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # (B, C, W)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn

        x1 = x
        

        # SRGA
        b, c, h, w = x.size() # (8, 256, 16, 16)
        theta_xs = self.theta_spatial(x)	# out_c = in_channel // cha_ratio  256 // 8 = 32  (8, 32, 16, 16)
        phi_xs = self.phi_spatial(x)        # (8, 32, 16, 16)
        theta_xs = theta_xs.view(b, self.inter_channel, -1) # (8, 32, 16 * 16) = (8, 32, 256)
        theta_xs = theta_xs.permute(0, 2, 1)  # (8, 256, 32)
        phi_xs = phi_xs.view(b, self.inter_channel, -1) # (8, 32, 16 * 16) = (8, 32, 256)
        Gs = torch.matmul(theta_xs, phi_xs) # (8, 256, 256)
        Gs_in = Gs.permute(0, 2, 1).view(b, h*w, h, w) # ->(8, 256, 256) -> (8, 256, 16, 16)
        Gs_out = Gs.view(b, h*w, h, w) # (8, 256, 16, 16)
        Gs_joint = torch.cat((Gs_in, Gs_out), 1) # (8, 512, 16, 16)
        Gs_joint = self.gg_spatial(Gs_joint) # -> in_spatial // spa_ratio  # 256 // 16 = 16 out_c = inter_sp = 16 -> (8, 16, 16, 16)
    
        g_xs = self.gx_spatial(x) # (8, 32, 16, 16)
        g_xs = torch.mean(g_xs, dim=1, keepdim=True) # (8, 1, 16, 16)
        ys = torch.cat((g_xs, Gs_joint), 1)  # (8, 16 + 1, 16, 16)

        W_ys = self.W_spatial(ys) # (8, 1, 16, 16)
        
        x = F.sigmoid(W_ys.expand_as(x)) * x  
        
        x = x1 + x

        # CSA
        # reduce calculations
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        # normalization first, then reshape -> (B, H, W, C) -> (B, C, H * W) and generate q, k and v
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x
