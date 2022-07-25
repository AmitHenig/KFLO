import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F

"""
Implementation Note:
When using our method, the weights of the convolution and fully connected layers are the reshaped result of 
a 1D convolution, and are calculated in each forward run. The Weight Decay value is applied on the calculated 
weights (to avoid tuning a hyper parameter). For this purpose a PyTorch amateur hack has been implement in the 
code where the L2 sum of the generated weights (for the whole network) can be received with the function get_kflo_l2.
"""


class KFLO(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=0, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,
                 bias=False, width_multiplier_r=4):
        super(KFLO, self).__init__()
        assert padding_mode == 'zeros', 'Only supports zero padding.'
        self.deploy = deploy
        self.l2_acb = 0

        w_ch = round(width_multiplier_r * out_channels)
        self.is_conv = kernel_size > 0
        self.is_bias = bias
        if self.is_conv:
            self.conv_stride = stride
            self.conv_padding = padding
            self.conv_dilation = dilation
            self.conv_groups = groups
            self.kflo_w1 = nn.Conv2d(in_channels=in_channels, out_channels=w_ch,
                                          kernel_size=(kernel_size, kernel_size), stride=stride,
                                          padding=padding, dilation=dilation, groups=groups, bias=bias,
                                          padding_mode=padding_mode)
        else:
            self.kflo_w1 = nn.Linear(in_features=in_channels, out_features=w_ch, bias=bias)

        self.kflo_w2 = nn.Conv2d(in_channels=w_ch, out_channels=out_channels,
                                     kernel_size=(1, 1), stride=1,
                                     padding=0, dilation=1, groups=1, bias=False,
                                     padding_mode='zeros')
        init.dirac_(self.kflo_w2.weight)

        self.w_orig_shape = list(self.kflo_w1.weight.shape)
        self.w_orig_shape[0] = out_channels
        self.w_cat_shape = list(self.kflo_w1.weight.shape)
        self.w_combine_shape = list(self.kflo_w2.weight.shape)

        prod_size = self.w_cat_shape[1] * self.w_cat_shape[2] * self.w_cat_shape[3] if self.is_conv else self.w_cat_shape[1]
        self.w_cat_1d_shape = [1, self.w_cat_shape[0], prod_size]
        self.w_combine_1d_shape = [self.w_combine_shape[0], self.w_combine_shape[1], 1]
        if self.kflo_w1.bias is not None:
            self.b_orig_shape = list(self.kflo_w1.bias.shape)
            self.b_orig_shape[0] = out_channels
            self.b_cat_1d_shape = [1, self.w_cat_shape[0], 1]

    def switch_to_deploy(self):
        self.deploy = True

    def calc_method_weights(self):
        w1 = self.kflo_w1.weight
        w1 = w1.view(*self.w_cat_1d_shape)
        w2 = self.kflo_w2.weight
        w2 = w2.view(*self.w_combine_1d_shape)
        w1 = F.conv1d(w1, w2)
        w1 = w1.view(self.w_orig_shape)
        self.l2_acb = torch.sum(w1 ** 2)
        if self.is_bias:
            b1 = self.kflo_w1.bias
            b1 = b1.view(*self.b_cat_1d_shape)
            b1 = F.conv1d(b1, w2)
            b1 = b1.view(self.b_orig_shape)
        else:
            b1 = None
        return w1, b1

    def forward(self, input):
        w1, b1 = self.calc_method_weights()
        if self.is_conv:
            result = F.conv2d(input, w1, bias=b1, stride=self.conv_stride, padding=self.conv_padding,
                              dilation=self.conv_dilation, groups=self.conv_groups)
        else:
            result = F.linear(input, w1, bias=b1)
        return result


def get_kflo_l2(net):
    method_wd_sum = 0
    for mmm in net.modules():
        if isinstance(mmm, KFLO):
            method_wd_sum += mmm.l2_acb
    return method_wd_sum


if __name__ == '__main__':
    N = 1
    C = 2
    H = 62
    W = 62
    O = 8
    groups = 4

    x = torch.randn(N, C)#, H, W)
    print('input shape is ', x.size())

    test_kernel_padding = [(3,1), (3,0), (5,1), (5,2), (5,3), (5,4), (5,6)]

    for k, p in [(3,1)]:  # test_kernel_padding:
        kflo = KFLO(C, O, kernel_size=k, padding=p, stride=1, deploy=False, bias=False)
        # kflo = KFLO(C, O, kernel_size=0, padding=p, stride=1, deploy=False, bias=True)
        kflo.eval()
        out = kflo(x)
        kflo.switch_to_deploy()
        deployout = kflo(x)
        print('difference between the outputs of the training-time and converted kflo is')
        print(((deployout - out) ** 2).sum())

