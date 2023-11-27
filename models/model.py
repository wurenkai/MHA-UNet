import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
import os
import sys
import torch.fft
import math

import traceback


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


if 'DWCONV_IMPL' in os.environ:
    try:
        sys.path.append(os.environ['DWCONV_IMPL'])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM


        def get_dwconv(dim, kernel, bias):
            return DepthWiseConv2dImplicitGEMM(dim, kernel, bias)
        # print('Using Megvii large kernel dw conv impl')
    except:
        print(traceback.format_exc())


        def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

        # print('[fail to use Megvii Large kernel] Using PyTorch large kernel dw conv impl')
else:
    def get_dwconv(dim, kernel, bias):
        return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

    # print('Using PyTorch large kernel dw conv impl')


class HA(nn.Module):
    """
    High-order Attention Interaction block
    Created on Wed Nov 02 10:14:25 2023
    @author: Renkai Wu
    Part of the code is based on the gnconv.
    """
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.sq = nn.ModuleList(
            [SqueezeAttentionBlock(self.dims[i], self.dims[i]) for i in range(order - 1)]
        )

        self.scale = s

        print('HA', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](self.sq[i](x)) * dw_list[i + 1]

        x = self.proj_out(x)

        return x


class HA_block(nn.Module):
    r""" HA-block block
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, HA=HA):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.HA = HA(dim)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.SA = SqueezeAttentionBlock(ch_in=dim, ch_out=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.HA(self.norm1(x)))

        input = x
        input = self.SA(input)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)

        return att1, att2, att3, att4


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4):
        t_list = [t1, t2, t3, t4]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3]


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4):
        r1, r2, r3, r4 = t1, t2, t3, t4

        satt1, satt2, satt3, satt4 = self.satt(t1, t2, t3, t4)
        t1, t2, t3, t4 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4

        r1_, r2_, r3_, r4_ = t1, t2, t3, t4
        t1, t2, t3, t4 = t1 + r1, t2 + r2, t3 + r3, t4 + r4

        catt1, catt2, catt3, catt4 = self.catt(t1, t2, t3, t4)
        t1, t2, t3, t4 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_


class MHA_UNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, pretrained=None,use_checkpoint=False, c_list=[16, 32, 64, 128, 256],
                 split_att='fc', bridge=True):
        super().__init__()
        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint
        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
            nn.Dropout2d(0.1),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
            MHAblock(c_list[2]))
        self.encoder3_1 = nn.Dropout2d(0.1)

        self.encoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[3], 3, stride=1, padding=1),
            MHAblock(c_list[3]))
        self.encoder4_1 = nn.Dropout2d(0.1)

        self.encoder5 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[4], 3, stride=1, padding=1),
            MHAblock(c_list[4]))
        self.encoder5_1 = nn.Dropout2d(0.1)

        # build Bottleneck layers
        self.ConvMixer = ConvMixerBlock(dim=c_list[4], depth=7, k=7)

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')


        self.decoder1 = nn.Sequential(
            nn.Dropout2d(0.1),
            MHAblock(c_list[4]))
        self.decoder1_1 = nn.Conv2d(c_list[4], c_list[3], 3, stride=1, padding=1)

        self.decoder2 = nn.Sequential(
            nn.Dropout2d(0.1),
            MHAblock(c_list[3]))
        self.decoder2_1 = nn.Conv2d(c_list[3], c_list[2], 3, stride=1, padding=1)

        self.decoder3 = nn.Sequential(
            nn.Dropout2d(0.1),
            MHAblock(c_list[2]))
        self.decoder3_1 = nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1)

        self.decoder4 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )


        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[3])
        self.dbn2 = nn.GroupNorm(4, c_list[2])
        self.dbn3 = nn.GroupNorm(4, c_list[1])
        self.dbn4 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3_1(self.encoder3(out)[0])), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4_1(self.encoder4(out)[0])), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        if self.bridge: t1, t2, t3, t4 = self.scab(t1, t2, t3, t4)
        out = F.gelu((self.ebn5(self.encoder5_1(self.encoder5(out)[0]))))# b, c4, H/32, W/32
        out = self.ConvMixer(out)

        out5 = F.gelu(self.dbn1(self.decoder1_1(self.decoder1(out)[0])))
        out5 = torch.add(out5, t4)

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2_1(self.decoder2(out5)[0])), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))
        out4 = torch.add(out4, t3)
        a = self.decoder3(out4)
        x1 = a[1]
        x2 = a[2]
        x3 = a[3]
        x4 = a[4]
        x5 = a[5]
        xx = a[0]

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3_1(xx)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))
        out3 = torch.add(out3, t2)

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))
        out2 = torch.add(out2, t1)

        out0 = F.interpolate(self.final(out2), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)

        return torch.sigmoid(out0),torch.sigmoid(x1),torch.sigmoid(x2),torch.sigmoid(x3),torch.sigmoid(x4),torch.sigmoid(x5)



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',
                                   align_corners=True).permute(1, 2, 3, 0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# bottleneck
class ConvMixerBlock(nn.Module):
    def __init__(self, dim=256, depth=7, k=7):
        super(ConvMixerBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(dim, dim, kernel_size=(k, k), groups=dim, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )

    def forward(self, x):
        x = self.block(x)
        return x

class EAblock(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, in_c, 1)

        self.k = in_c * 4
        self.linear_0 = nn.Conv1d(in_c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, in_c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

        self.conv2 = nn.Conv2d(in_c, in_c, 1, bias=False)
        self.norm_layer = nn.GroupNorm(4, in_c)

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n

        attn = self.linear_0(x)  # b, k, n
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
        x = self.linear_1(attn)  # b, c, n

        x = x.view(b, c, h, w)
        x = self.norm_layer(self.conv2(x))
        x = x + idn
        x = F.gelu(x)
        return x

class HA1(nn.Module):
    def __init__(self, c1, HA=HA, block=HA_block):
        super(HA1, self).__init__()


        if not isinstance(HA, list):
            HA = [partial(HA, order=1, s=1 / 3),
                  partial(HA, order=2, s=1 / 3),
                  partial(HA, order=3, s=1 / 3),
                  partial(HA, order=4, s=1 / 3, h=24, w=13, gflayer=GlobalLocalFilter),
                  partial(HA, order=5, s=1 / 3, h=12, w=7, gflayer=GlobalLocalFilter)]
        else:
            HA = HA
            assert len(HA) == 5

        if isinstance(HA[0], str):
            HA = [eval(H) for H in HA]

        if isinstance(block, str):
            block = eval(block)

        self.HA = nn.Sequential(
            EAblock(c1),
            *[block(dim=c1, drop_path=0.4,
                    layer_scale_init_value=1e-6, HA=HA[0]) for j in range(1)],
        )
    def forward(self, x):
        return self.HA(x)

    def forward_fuse(self, x):
        return self.HA(x)

class HA2(nn.Module):
    def __init__(self, c1, HA=HA, block=HA_block):
        super(HA2, self).__init__()


        if not isinstance(HA, list):
            HA = [partial(HA, order=1, s=1 / 3),
                  partial(HA, order=2, s=1 / 3),
                  partial(HA, order=3, s=1 / 3),
                  partial(HA, order=4, s=1 / 3, h=24, w=13, gflayer=GlobalLocalFilter),
                  partial(HA, order=5, s=1 / 3, h=12, w=7, gflayer=GlobalLocalFilter)]
        else:
            HA = HA
            assert len(HA) == 5

        if isinstance(HA[0], str):
            HA = [eval(H) for H in HA]

        if isinstance(block, str):
            block = eval(block)

        self.HA = nn.Sequential(
            EAblock(c1),
            *[block(dim=c1, drop_path=0.4,
                    layer_scale_init_value=1e-6, HA=HA[1]) for j in range(1)],
        )
    def forward(self, x):
        return self.HA(x)

    def forward_fuse(self, x):
        return self.HA(x)

class HA3(nn.Module):
    def __init__(self, c1, HA=HA, block=HA_block):
        super(HA3, self).__init__()


        if not isinstance(HA, list):
            HA = [partial(HA, order=1, s=1 / 3),
                  partial(HA, order=2, s=1 / 3),
                  partial(HA, order=3, s=1 / 3),
                  partial(HA, order=4, s=1 / 3, h=24, w=13, gflayer=GlobalLocalFilter),
                  partial(HA, order=5, s=1 / 3, h=12, w=7, gflayer=GlobalLocalFilter)]
        else:
            HA = HA
            assert len(HA) == 5

        if isinstance(HA[0], str):
            HA = [eval(H) for H in HA]

        if isinstance(block, str):
            block = eval(block)

        self.HA = nn.Sequential(
            EAblock(c1),
            *[block(dim=c1, drop_path=0.4,
                    layer_scale_init_value=1e-6, HA=HA[2]) for j in range(1)],
        )
    def forward(self, x):
        return self.HA(x)

    def forward_fuse(self, x):
        return self.HA(x)

class HA4(nn.Module):
    def __init__(self, c1, HA=HA, block=HA_block):
        super(HA4, self).__init__()


        if not isinstance(HA, list):
            HA = [partial(HA, order=1, s=1 / 3),
                  partial(HA, order=2, s=1 / 3),
                  partial(HA, order=3, s=1 / 3),
                  partial(HA, order=4, s=1 / 3, h=24, w=13, gflayer=GlobalLocalFilter),
                  partial(HA, order=5, s=1 / 3, h=12, w=7, gflayer=GlobalLocalFilter)]
        else:
            HA = HA
            assert len(HA) == 5

        if isinstance(HA[0], str):
            HA = [eval(H) for H in HA]

        if isinstance(block, str):
            block = eval(block)

        self.HA = nn.Sequential(
            EAblock(c1),
            *[block(dim=c1, drop_path=0.4,
                    layer_scale_init_value=1e-6, HA=HA[3]) for j in range(1)],
        )
    def forward(self, x):
        return self.HA(x)

    def forward_fuse(self, x):
        return self.HA(x)

class HA5(nn.Module):
    def __init__(self, c1, HA=HA, block=HA_block):
        super(HA5, self).__init__()


        if not isinstance(HA, list):
            HA = [partial(HA, order=1, s=1 / 3),
                  partial(HA, order=2, s=1 / 3),
                  partial(HA, order=3, s=1 / 3),
                  partial(HA, order=4, s=1 / 3, h=24, w=13, gflayer=GlobalLocalFilter),
                  partial(HA, order=5, s=1 / 3, h=12, w=7, gflayer=GlobalLocalFilter)]
        else:
            HA = HA
            assert len(HA) == 5

        if isinstance(HA[0], str):
            HA = [eval(H) for H in HA]

        if isinstance(block, str):
            block = eval(block)

        self.HA = nn.Sequential(
            EAblock(c1),
            *[block(dim=c1, drop_path=0.4,
                    layer_scale_init_value=1e-6, HA=HA[4]) for j in range(1)],
        )
    def forward(self, x):
        return self.HA(x)

    def forward_fuse(self, x):
        return self.HA(x)

class MHAblock(nn.Module):
    """
    Multiple High-order Attention Interaction block
    Created on Wed Nov 02 10:14:25 2023
    @author: Renkai Wu
    """

    def __init__(self, channel):
        super(MHAblock, self).__init__()
        self.channel = channel
        self.g1c = nn.Sequential(
            HA1(channel),
            nn.BatchNorm2d(self.channel),
        )
        self.g2c = nn.Sequential(
            HA2(channel),
            nn.BatchNorm2d(self.channel),
        )
        self.g3c = nn.Sequential(
            HA3(channel),
            nn.BatchNorm2d(self.channel),
        )
        self.g4c = nn.Sequential(
            HA4(channel),
            nn.BatchNorm2d(self.channel),
        )
        self.g5c = nn.Sequential(
            HA5(channel),
            nn.BatchNorm2d(self.channel),
        )
        self.voteConv = nn.Sequential(
            SqueezeAttentionBlock(ch_in=self.channel * 5, ch_out=self.channel * 5),
            nn.Conv2d(self.channel * 5, self.channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.g1c(x)
        x1 = self.g2c(x)
        x2 = self.g3c(x)
        x3 = self.g4c(x)
        x4 = self.g5c(x)
        _x = self.relu(torch.cat((x0, x1, x2, x3, x4), dim=1))
        a =_x
        _x = self.voteConv(_x)
        x = x + x * _x
        return x,x0,x1,x2,x3,x4,a,_x

class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(ch_in, ch_out)
        self.conv_atten = conv_block(ch_in, ch_out)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        #print("x.shape: ", x.shape)
        x_res = self.conv(x)
        #print("x_res.shape: ", x_res.shape)
        y = self.avg_pool(x)
        #print("y.shape dopo avg pool: ", y.shape)
        y = self.conv_atten(y)
        #print("y.shape dopo conv att:", y.shape)
        y = self.upsample(y)
        #print(y.shape, x_res.shape)
        #print("(y * x_res) + y: ", (y * x_res) + y)
        return (y * x_res) + y

def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
