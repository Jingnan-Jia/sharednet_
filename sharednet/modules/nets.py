# -*- coding: utf-8 -*-
# @Time    : 11/20/20 10:39 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
from typing import Sequence, Union

import torch
import torch.nn as nn
import numpy as np
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep
from monai.networks.nets.basic_unet import TwoConv, Down, UpCat



def model_summary(model, model_name):
    print(f"=================model_summary: {model_name}==================")
    # model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Total Params:{params}")
    print("=" * 100)


class UpCatConvCond(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple] = 0.1,
        upsample: str = "deconv",
        halves: bool = True,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            halves: whether to halve the number of channels during upsampling.
        """
        super().__init__()

        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(dim, in_chns, up_chns, 2, mode=upsample)
        self.convs = TwoConv(dim, cat_chns + up_chns, out_chns, act, norm, dropout)
        self.convs_cond = TwoConv(dim, cat_chns + up_chns + 1, out_chns, act, norm, dropout)

    def forward(self, x: torch.Tensor, x_e: torch.Tensor, cond: torch.Tensor):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
            cond: features from the conditioning filter. It's channel number is only 1.
        """
        x_0 = self.upsample(x)
        # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
        dimensions = len(x.shape) - 2
        sp = [0] * (dimensions * 2)
        for i in range(dimensions):
            if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                sp[i * 2 + 1] = 1
        x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
        if cond is None:
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            ones = torch.ones(x_0.shape[0], 1, *x_0.shape[2:]).to(torch.device("cuda"))
            cond = ones * cond[:, None, None, None, None].to(torch.device("cuda"))
            x = self.convs_cond(torch.cat([x_e, x_0, cond], dim=1))  # input channels: (cat_chns + up_chns + 1)
        return x


class UpCatConv(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple] = 0.1,
        upsample: str = "deconv",
        halves: bool = True,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            halves: whether to halve the number of channels during upsampling.
        """
        super().__init__()

        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(dim, in_chns, up_chns, 2, mode=upsample)
        self.convs = TwoConv(dim, cat_chns + up_chns, out_chns, act, norm, dropout)

    def forward(self, x: torch.Tensor, x_e: torch.Tensor):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
            cond: features from the conditioning filter. It's channel number is only 1.
        """
        x_0 = self.upsample(x)

        # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
        dimensions = len(x.shape) - 2
        sp = [0] * (dimensions * 2)
        for i in range(dimensions):
            if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                sp[i * 2 + 1] = 1
        x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
        x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        return x


class ConvCond(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple] = 0.1,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            halves: whether to halve the number of channels during upsampling.
        """
        super().__init__()

        self.convs_cond = TwoConv(dim, in_chns + 1, out_chns, act, norm, dropout)
        self.convs = TwoConv(dim, in_chns, out_chns, act, norm, dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """

        Args:
            x: input features.
            cond: conditioning features.
        """
        if cond == None:  # No conditioning
            x = self.convs(x)  # input channels: in_chns
        else:
            ones = torch.ones(x.shape[0], 1, *x.shape[2:]).to(torch.device("cuda"))
            cond =  ones * cond[:, None, None, None, None].to(torch.device("cuda"))
            x = self.convs_cond(torch.cat([x, cond], dim=1))  # input channels: (in_chns + 1)
        return x


class CondnetMulAdd(nn.Module):
    def __init__(
            self,
            dimensions: int = 3,
            in_channels: int = 1,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("batch", {"affine": True}),
            dropout: Union[float, tuple] = 0.1,

            out_channels: int = 1,
            upsample: str = "deconv",
            cond_pos: str = "enc"

    ):
        """
               A UNet implementation with 1D/2D/3D supports.

               Based on:

                   Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
                   Morphometry". Nature Methods 16, 67–70 (2019), DOI:
                   http://dx.doi.org/10.1038/s41592-018-0261-2

               Args:
                   dimensions: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
                   in_channels: number of input channels. Defaults to 1.
                   features: six integers as numbers of features.
                       Defaults to ``(32, 32, 64, 128, 256, 32)``,

                       - the first five values correspond to the five-level encoder feature sizes.
                       - the last value corresponds to the feature size after the last upsampling.

                   act: activation type and arguments. Defaults to LeakyReLU.
                   norm: feature normalization type and arguments. Defaults to instance norm.
                   dropout: dropout ratio. Defaults to no dropout.
                   upsample: upsampling mode, available options are
                       ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

               Examples::

               See Also

                   - :py:class:`monai.networks.nets.DynUNet`
                   - :py:class:`monai.networks.nets.UNet`

               """
        super().__init__()
        self.cond_pos = cond_pos
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")
        self.max_pooling = Pool["MAX", dimensions](kernel_size=2)

        self.conv_0 = ConvCond(dimensions, in_channels, features[0], act, norm, dropout)
        self.conv_1 = ConvCond(dimensions, fea[0], fea[1], act, norm, dropout)
        self.conv_2 = ConvCond(dimensions, fea[1], fea[2], act, norm, dropout)
        self.conv_3 = ConvCond(dimensions, fea[2], fea[3], act, norm, dropout)
        self.conv_4 = ConvCond(dimensions, fea[3], fea[4], act, norm, dropout)

        self.upcat_4 = UpCatConv(dimensions, fea[4], fea[3], fea[3], act, norm, dropout, upsample)
        self.upcat_3 = UpCatConv(dimensions, fea[3], fea[2], fea[2], act, norm, dropout, upsample)
        self.upcat_2 = UpCatConv(dimensions, fea[2], fea[1], fea[1], act, norm, dropout, upsample)
        self.upcat_1 = UpCatConv(dimensions, fea[1], fea[0], fea[5], act, norm, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", dimensions](fea[5], out_channels, kernel_size=1)

    def forward(self, x_batch: torch.Tensor, class_id_batch: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.
            class_id_batch: a batch of class_id

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        condition: torch.Tensor = class_id_batch
        if self.cond_pos == 'input':
            condition_input = condition
            condition_enc, condition_dec = None, None
        elif self.cond_pos == 'enc':
            condition_input, condition_dec = None, None
            condition_enc = condition
        elif self.cond_pos == 'dec':
            condition_input, condition_enc = None, None
            condition_dec = condition
        elif self.cond_pos == 'enc_dec':
            condition_dec, condition_enc = condition, condition
            condition_input = None
        else:
            raise Exception(f"cond_pos is wrong: {self.cond_pos}")

        x0 = self.conv_0(x_batch, condition_input)
        x0_down = self.max_pooling(x0)

        x1 = self.conv_1(x0_down, condition_enc)
        x1_down = self.max_pooling(x1)

        x2 = self.conv_2(x1_down, condition_enc)
        x2_down = self.max_pooling(x2)

        x3 = self.conv_3(x2_down, condition_enc)
        x3_down = self.max_pooling(x3)

        x4 = self.conv_4(x3_down, condition_enc)

        u3 = self.upcat_4(x4, x3, condition_dec)
        u2 = self.upcat_3(u3, x2, condition_dec)
        u1 = self.upcat_2(u2, x1, condition_dec)
        u0 = self.upcat_1(u1, x0, condition_dec)

        u_out = self.final_conv(u0)

        return u_out


class CondnetConcat(nn.Module):
    def __init__(
            self,
            dimensions: int = 3,
            in_channels: int = 1,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("batch", {"affine": True}),
            dropout: Union[float, tuple] = 0.1,

            out_channels: int = 1,
            upsample: str = "deconv",
            cond_pos: str = "enc"

    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            dimensions: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        Examples::


        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        self.cond_pos = cond_pos
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")
        self.max_pooling = Pool["MAX", dimensions](kernel_size=2)

        self.conv_0 = ConvCond(dimensions, in_channels, fea[0], act, norm, dropout)
        self.conv_1 = ConvCond(dimensions, fea[0], fea[1], act, norm, dropout)
        self.conv_2 = ConvCond(dimensions, fea[1], fea[2], act, norm, dropout)
        self.conv_3 = ConvCond(dimensions, fea[2], fea[3], act, norm, dropout)
        self.conv_4 = ConvCond(dimensions, fea[3], fea[4], act, norm, dropout)


        self.upcat_4 = UpCatConvCond(dimensions, fea[4], fea[3], fea[3], act, norm, dropout, upsample)
        self.upcat_3 = UpCatConvCond(dimensions, fea[3], fea[2], fea[2], act, norm, dropout, upsample)
        self.upcat_2 = UpCatConvCond(dimensions, fea[2], fea[1], fea[1], act, norm, dropout, upsample)
        self.upcat_1 = UpCatConvCond(dimensions, fea[1], fea[0], fea[5], act, norm, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", dimensions](fea[5], out_channels, kernel_size=1)

    def forward(self, x_batch: torch.Tensor, class_id_batch: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.
            class_id: a batch of inters or a batch of None.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        condition: torch.Tensor = class_id_batch
        if self.cond_pos == 'input':
            condition_input = condition
            condition_enc, condition_dec = None, None
        elif self.cond_pos == 'enc':
            condition_input, condition_dec = None, None
            condition_enc = condition
        elif self.cond_pos == 'dec':
            condition_input, condition_enc = None, None
            condition_dec = condition
        elif self.cond_pos == 'enc_dec':
            condition_dec, condition_enc = condition, condition
            condition_input = None
        else:
            raise Exception(f"cond_pos is wrong: {self.cond_pos}")

        x0 = self.conv_0(x_batch, condition_input)
        x0_down = self.max_pooling(x0)

        x1 = self.conv_1(x0_down, condition_enc)
        x1_down = self.max_pooling(x1)

        x2 = self.conv_2(x1_down, condition_enc)
        x2_down = self.max_pooling(x2)

        x3 = self.conv_3(x2_down, condition_enc)
        x3_down = self.max_pooling(x3)

        x4 = self.conv_4(x3_down, condition_enc)

        u3 = self.upcat_4(x4, x3, condition_dec)
        u2 = self.upcat_3(u3, x2, condition_dec)
        u1 = self.upcat_2(u2, x1, condition_dec)
        u0 = self.upcat_1(u1, x0, condition_dec)

        u_out = self.final_conv(u0)

        return u_out


def condnet(cond_method, cond_pos, out_chn, base):

    if cond_method=='concat':
        net = CondnetConcat(out_channels=out_chn,
                            features=(base, base, 2 * base, 4 * base, 8 * base, base),
                            dropout=0.1,  # when chaning it, do not forget to update log_param as well!
                            cond_pos=cond_pos)
    elif cond_method=='mul_add':
        net = CondnetMulAdd(out_channels=out_chn,
                            features=(base, base, 2 * base, 4 * base, 8 * base, base),
                            dropout=0.1,
                            cond_pos=cond_pos)
    else:
        raise Exception(f"cond_method should be 'concat' or 'mul_add', but the current valueis {cond_method}.")
    return net




class UNet(nn.Module):
    def __init__(
            self,
            dimensions: int = 3,
            in_channels: int = 1,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("batch", {"affine": True}),
            dropout: Union[float, tuple] = 0.1,
            out_channels: int = 1,
            upsample: str = "deconv",

    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            dimensions: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        Examples::

        """
        super().__init__()
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")
        self.max_pooling = Pool["MAX", dimensions](kernel_size=2)

        self.conv_0 = TwoConv(dimensions, in_channels, fea[0], act, norm, dropout)
        self.conv_1 = TwoConv(dimensions, fea[0], fea[1], act, norm, dropout)
        self.conv_2 = TwoConv(dimensions, fea[1], fea[2], act, norm, dropout)
        self.conv_3 = TwoConv(dimensions, fea[2], fea[3], act, norm, dropout)
        self.conv_4 = TwoConv(dimensions, fea[3], fea[4], act, norm, dropout)

        self.upcat_4 = UpCatConv(dimensions, fea[4], fea[3], fea[3], act, norm, dropout, upsample)
        self.upcat_3 = UpCatConv(dimensions, fea[3], fea[2], fea[2], act, norm, dropout, upsample)
        self.upcat_2 = UpCatConv(dimensions, fea[2], fea[1], fea[1], act, norm, dropout, upsample)
        self.upcat_1 = UpCatConv(dimensions, fea[1], fea[0], fea[5], act, norm, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", dimensions](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.
            cond: it is only as a placeholder so that I do not need to custom new dataloader output and evaluator

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """

        x0 = self.conv_0(x)
        x0_down = self.max_pooling(x0)

        x1 = self.conv_1(x0_down)
        x1_down = self.max_pooling(x1)

        x2 = self.conv_2(x1_down)
        x2_down = self.max_pooling(x2)

        x3 = self.conv_3(x2_down)
        x3_down = self.max_pooling(x3)

        x4 = self.conv_4(x3_down)

        u3 = self.upcat_4(x4, x3)
        u2 = self.upcat_3(u3, x2)
        u1 = self.upcat_2(u2, x1)
        u0 = self.upcat_1(u1, x0)

        u_out = self.final_conv(u0)

        return u_out



def get_net(cond_flag, cond_method, cond_pos, out_chn, base):
    if cond_flag:
        net = condnet(cond_method=cond_method, cond_pos=cond_pos, out_chn=out_chn, base=base) # base=32
    else:
        net = UNet(features = (base * 1, base * 1, base * 2, base * 4, base * 8, base), out_channels=out_chn)

    return net