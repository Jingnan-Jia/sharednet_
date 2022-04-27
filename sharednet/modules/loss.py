
import monai
import torch
from torch import nn as nn
from torch.nn.modules.loss import _Loss
from typing import Callable, List, Optional, Sequence, Union
from monai.utils import LossReduction, Weight
import warnings
from monai.networks import one_hot


class WeightedDiceLoss(_Loss):
    """Weighted Dice for multi-class segmentation. Small objects would be bigger weights.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = True,
        sigmoid: bool = False,
        softmax: bool = True,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if setted)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)


        volume_tot = torch.sum(ground_o)
        ratio_per_class = ground_o / volume_tot
        weight_per_class = 1 / (ratio_per_class + self.smooth_nr)
        weight_tot = torch.sum(weight_per_class)
        normalized_weight_per_class = weight_per_class / weight_tot
        print(f'normalize weights: {normalized_weight_per_class}')
        f = f * normalized_weight_per_class
        f = torch.mean(f)  # the batch and channel average

        return f


class WeightedCELoss(nn.Module):
    """
    this is soft CrossEntropyLoss
    """

    def __init__(self, mode='fnfp', adap_weights=None):

        super().__init__()
        # self.nllloss = nn.NLLLoss()
        # self.softmax = nn.functional.softmax(dim=1)
        # return
        self.mode = mode
        self.adap_weights = adap_weights

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        # loss = nn.CrossEntropyLoss()
        # celoss = loss(input,target.long())

        target = target.unsqueeze(1)
        # print(input.shape)
        # print(target.shape)
        batch = input.shape[0]
        cls = input.shape[1]
        target_onehot = torch.FloatTensor(input.shape)
        if target.device.type == "cuda":
            target_onehot = target_onehot.cuda(target.device.index)
        target_onehot.zero_()
        target_onehot.scatter_(1, target.type(torch.int64), 1)
        if self.adap_weights == None:
            reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
            self.weights = torch.sum(target_onehot, dim=reduce_axis)
            self.weights = self.weights / torch.sum(self.weights)

            self.weights = 1 / self.weights
            self.weights = self.weights / torch.sum(self.weights)

            print(f'weights for CE: {self.weights}')
        input_pro = nn.functional.softmax(input, dim=1)
        fn_logpro = torch.log(input_pro + 1e-6)
        fp_logpro = torch.log(1 - input_pro + 1e-6)

        ce_fn = -1 * fn_logpro * target_onehot
        ce_fn = ce_fn.view(batch, cls, -1)
        ce_fn = torch.mean(ce_fn, dim=2)
        ce_fn = self.weights * ce_fn
        ce_fn = torch.sum(ce_fn, dim=1)
        ce_fn = torch.mean(ce_fn, dim=0)
        if self.mode == 'fn':
            return ce_fn
        else:
            ce_fp = -1 * fp_logpro * (1 - target_onehot)
            ce_fp = ce_fp.view(batch, cls, -1)
            ce_fp = torch.mean(ce_fp, dim=2)
            ce_fp = self.weights * ce_fp
            ce_fp = torch.sum(ce_fp, dim=1)
            ce_fp = torch.mean(ce_fp, dim=0)

            return ce_fn + ce_fp


class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        print(f"dice loss: {dice}, CE loss: {cross_entropy}")
        return dice + cross_entropy


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return cross_entropy


def get_loss(loss: str = "dice") -> nn.Module:
    """Return loss function from its name.

    Args:
        task: task name

    """
    loss_fun: nn.Module

    if loss == "dice":
        loss_fun = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
    elif loss == "CE":
        loss_fun = CELoss()
    elif loss == "dice_CE":
        loss_fun = DiceCELoss()  # or FocalLoss
    elif loss == "weighted_dice":
        loss_fun = WeightedDiceLoss(to_onehot_y=True, softmax=True)
    elif loss == "weighted_CE_fnfp":
        loss_fun = WeightedCELoss()
    elif loss == "weighted_CE_fn":
        loss_fun = WeightedCELoss(mode='fn')
    else:
        raise ValueError(f"loss_fun should be 'dice', 'CE', 'dice_CE', 'weighted_dice', 'weighted_CE', but got {loss}")
    return loss_fun