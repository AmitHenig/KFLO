from builder import ConvBuilder
from kflo.kflo_block import KFLO


class KFLOBuilder(ConvBuilder):

    def __init__(self, base_config, deploy):
        super(KFLOBuilder, self).__init__(base_config=base_config)
        self.deploy = deploy
        self.use_last_bn = False

    def switch_to_deploy(self):
        self.deploy = True

    def Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', use_original_conv=False):
        return KFLO(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                       dilation=dilation, groups=groups, padding_mode=padding_mode, deploy=self.deploy, bias=bias)

    def Linear(self, in_features, out_features, bias=True):
        return KFLO(in_features, out_features, deploy=self.deploy, bias=bias)

    def IntermediateLinear(self, in_features, out_features, bias=True):
        return KFLO(in_features, out_features, deploy=self.deploy, bias=bias)
