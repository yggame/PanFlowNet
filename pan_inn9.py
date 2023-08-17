from math import exp
import torch
import torch.nn as nn
import torch.nn.init as init


class HinResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(HinResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(feature // 2, affine=True)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))

        out_1, out_2 = torch.chunk(residual, 2, dim=1)
        residual = torch.cat([self.norm(out_1), out_2], dim=1)

        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)




def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


# class CALayer(nn.Module):
#     def __init__(self, channel, reduction):
#         super(CALayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.contrast = stdv_channels
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#         self.process = nn.Sequential(
#             nn.Conv2d(channel, channel, 3, stride=1, padding=1),
#             nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(channel, channel, 3, stride=1, padding=1)
#         )

#         initialize_weights([self.conv_du,self.process],0.1)

#     def forward(self, x):
#         y = self.process(x)
#         y = self.avg_pool(y)+self.contrast(y)
#         z = self.conv_du(y)
#         return z * y + x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        initialize_weights([self.conv_du],0.1)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class HIN_CA_Block(nn.Module):
    def __init__(self, channel_in, channel_out) -> None:
        super().__init__()
        self. hin = HinResBlock(channel_in, channel_out)
        self.ca = CALayer(channel_out, reduction=4)

    def forward(self, x):
        hin_out = self.hin(x)
        ca_out = self.ca(hin_out)
        return ca_out


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        elif net_structure == 'Resnet':
            return ResBlock(channel_in, channel_out)
        elif net_structure == 'HinResnet':
            return HinResBlock(channel_in, channel_out)
        elif net_structure == 'HIN_CA':
            return HIN_CA_Block(channel_in, channel_out)
        else:
            return None
    return constructor


class subnet_coupling_layer(nn.Module):
    def __init__(self, dims_in, F_class, condition_length, clamp=5.):
        super().__init__()

        channels = 4
        self.ndims = 3
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.conditional = True
        # self.subnet = subnet

        self.s1 = F_class(self.split_len1 + condition_length, self.split_len2*2)
        self.s2 = F_class(self.split_len2 + condition_length, self.split_len1*2)

        # self.sigmoid = nn.Sigmoid()

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, cond1, cond2, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        # c_star = self.subnet(torch.cat(c, 1))

        if not rev:
            r2 = self.s2(torch.cat([x2, cond1, cond2], 1) if self.conditional else x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(torch.cat([y1, cond1, cond2], 1) if self.conditional else y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1
            self.last_jac = self.log_e(s1) + self.log_e(s2)

        else: # names of x and y are swapped!
            r1 = self.s1(torch.cat([x1, cond1, cond2], 1) if self.conditional else x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(torch.cat([y2, cond1, cond2], 1) if self.conditional else y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)
            self.last_jac = - self.log_e(s1) - self.log_e(s2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, rev=False):
        return torch.sum(self.last_jac, dim=tuple(range(1, self.ndims+1)))

    def output_dims(self, input_dims):
        return input_dims
    

class pan_inn(nn.Module):
    def __init__(self, mid_channels=64, T = 4):
        super().__init__()

        print("now: pan_inn9")
        self.up_factor = 4
        G0 = mid_channels
        kSize = 3

        # T = 4
        # todo
        operations = []
        b = subnet_coupling_layer(dims_in=4 , F_class=subnet('HIN_CA'), condition_length=5)
        for _ in range(T):
            # b = subnet_coupling_layer(dims_in=4 , F_class=subnet('HIN_CA'), condition_length=5)
            operations.append(b)
        
        self.operations = nn.ModuleList(operations)

    def forward(self, lms, b_ms, pan, rev=False):
        hms = torch.nn.functional.interpolate(lms, scale_factor=self.up_factor, mode='bilinear', align_corners=False)   # B 4 256 256
        x = hms
        b, c, h, w = x.shape

        # if cuda
        # out = torch.cuda.FloatTensor(x.shape).normal_(torch.mean(x), torch.std(x))
        # noise = out
        # out = out.detach() / out.detach().sum()
        neg_log = []
        # else
        # out = torch.FloatTensor(x.shape).normal_(torch.mean(x), torch.std(x))
        # out = out / out.sum()
        if not rev: 
            out = b_ms
            for op in self.operations:
                out = op(out, x, pan, rev)
                out2 = torch.sum(torch.sum(out**2, dim=1),dim=(1,2))
                jac = op.jacobian()
                neg_log_likeli = 0.5 * out2 - jac
                # jac_loss = torch.mean(neg_log_likeli)/(2 * h * w)
                neg_log.append(neg_log_likeli)
        else:
            out = torch.cuda.FloatTensor(x.shape).normal_(torch.mean(x), torch.std(x))
            # noise = out
            for op in reversed(self.operations):
                out = op(out, x, pan, rev)
                out2 = torch.sum(torch.sum(out**2, dim=1),dim=(1,2))
                jac = op.jacobian(rev=rev)
                neg_log_likeli = 0.5 * out2 - jac
                # jac_loss = torch.mean(neg_log_likeli)
                neg_log.append(neg_log_likeli)
        jac_loss = torch.mean(sum(neg_log))/(2 * h * w)
        return out, jac_loss#, noise

    def test(self, device='cpu'):
        total_params = sum(p.numel() for p in self.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        input_ms = torch.rand(4, 4, 64, 64)    # 196 为在Embeddings中的 n_patches （14 * 14）
        input_pan = torch.rand(4, 1, 256, 256)
        
        ideal_out = torch.rand(4, 4, 256, 256)
        
        out, _ = self.forward(input_ms, None, input_pan)
        
        assert out.shape == ideal_out.shape
        # import torchsummaryX
        # torchsummaryX.summary(self, input_ms.to(device), None, input_pan.to(device))

        #
        # from thop import profile
        # flops, params = profile(self, inputs=(input_ms,None, input_pan))
        # print("flops:", flops/1e9, "G")
        # print("params:", params/1e6, "M")

if __name__ == "__main__":


    net3 = pan_inn(T=4)
    net3.test()
