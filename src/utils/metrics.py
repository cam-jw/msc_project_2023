import torch
import numpy as np
from typing import List, Tuple, Dict
from sklearn.calibration import calibration_curve
import torchvision
import torchio as tio

def get_variance(
        stack: torch.Tensor
        ) -> torch.Tensor:
    return torch.nan_to_num(torch.var(stack, dim=0))

def get_mean(
        stack: torch.Tensor
        ) -> torch.Tensor:
    return torch.mean(stack, dim=0)

def get_predictive_entropy(
        mean: torch.Tensor
        ) -> torch.Tensor:
    return torch.nan_to_num(-mean * torch.log(mean))

def get_predictive_variance(
        mean: torch.Tensor
    ):
    return torch.nan_to_num(mean * (1 - mean))

def get_mutual_information(
        predictive_entropy: torch.Tensor, 
        expected_entropy: torch.Tensor
    ) -> torch.Tensor:
    return predictive_entropy - expected_entropy

def get_epistemic_variance(
        stack: torch.Tensor,
        mean: torch.Tensor
    ) -> torch.Tensor:
    mean = torch.repeat_interleave(mean, repeats=stack.shape[0], dim=0)[:, None, :, :, :]
    return torch.mean((stack - mean)**2, dim=0)

def get_expected_entropy(
        stack: torch.Tensor
    ) -> torch.Tensor:
    return torch.mean(torch.nan_to_num(-stack * torch.log(stack)), dim=0)

def get_aleatoric_variance(
        stack: torch.Tensor
    ) -> torch.Tensor:
    return torch.mean(torch.nan_to_num(stack * (1 - stack)), dim=0) 

def downsize(image, kernel_size=16, mask=False, mp=False):
    new_size = int(image.shape[-1] / kernel_size)
    if kernel_size > 1:
        if mask:
            new = torchvision.transforms.functional.resize(
                image, 
                (new_size, new_size),
                antialias=True
                )
            new[new >= 0.5] = 1.
            new[new < 0.5] = 0.
        else:
            new = torchvision.transforms.functional.resize(
                image, 
                (new_size, new_size),
                antialias=True
                )
            new = torch.nn.functional.max_pool2d(new, kernel_size=3, stride=1, padding=1)
        return new
    else:
        return image
    

def get_all_uncertainties(stack: torch.Tensor, kernel_size, mp: bool = True):
    out_dict = {}
    mean = get_mean(stack)
    predictive_entropy = get_predictive_entropy(mean)
    predictive_variance = get_predictive_variance(mean)

    expected_entropy = get_expected_entropy(stack)
    aleatoric_variance = get_aleatoric_variance(stack)

    mutual_information = get_mutual_information(predictive_entropy, expected_entropy)
    epistemic_variance = get_epistemic_variance(stack, mean)

    out_dict["predictive_entropy"] = downsize(predictive_entropy, kernel_size, mask=False, mp=mp)
    out_dict["predictive_variance"] = downsize(predictive_variance, kernel_size, mask=False, mp=mp)
    out_dict["expected_entropy"] = downsize(expected_entropy, kernel_size, mask=False, mp=mp)
    out_dict["aleatoric_variance"] = downsize(aleatoric_variance, kernel_size, mask=False, mp=mp)
    out_dict["mutual_information"] = downsize(mutual_information, kernel_size, mask=False, mp=mp)
    out_dict["epistemic_variance"] = downsize(epistemic_variance, kernel_size, mask=False, mp=mp)

    del predictive_entropy
    del predictive_variance
    del expected_entropy
    del aleatoric_variance
    del mutual_information
    del epistemic_variance

    return mean, out_dict

def mine(section):
    current_mean = section.mean()
    current_variance = section.var()

    # Calculate scaling factor and shifting value
    desired_mean = 0.5
    desired_variance = 0.05
    scale_factor = (desired_variance / current_variance) ** 0.5
    shift_value = desired_mean - (current_mean * scale_factor)

    # Normalize images
    normalized_images = (section * scale_factor) + shift_value

    return normalized_images

    
def normalise(
        section: torch.Tensor, 
        constraint: str = None
        ) -> torch.Tensor:
    orig_device = section.device
    if constraint == "abs":
        section = torch.abs(section)
    elif constraint == "shift_up":
        section = section + torch.min(section)
    elif constraint == "half":
        section = section / torch.max(section)
        section = abs(section - 0.5)
    elif constraint == "rescale":
        rescaler = tio.RescaleIntensity(
            out_min_max=(0, 1)
        )
        section = rescaler(section.cpu()).to(orig_device)
        return section
    elif constraint == "z":
        rescaler = tio.ZNormalization()
        section = rescaler(section[:, 0, :, :, :].cpu())[:, None, :, :, :].to(orig_device)
    elif constraint == "mine":
        section = mine(section)
        section = section / torch.max(section)
    elif constraint == "sigmoid":
        section = torch.nn.functional.sigmoid(section)
    return (section - torch.min(section)) / (torch.max(section) - torch.min(section))

def threshold_above_below(
        section: torch.Tensor, 
        value: float
) -> Tuple[torch.Tensor]:
    above = (section >= value).float()
    below = (section < value).float()
    return above, below


def threshold(
        section: torch.Tensor,
        value: float
) -> torch.Tensor:
    return (section >= value).float()


def get_binary_accurate_inaccurate(
        section: torch.Tensor, 
        label: torch.Tensor
) -> Tuple[torch.Tensor]:
    accurate = (section == label).float()
    inaccurate = (section != label).float()
    return accurate, inaccurate


def get_accuracy(
        prediction: torch.Tensor,
        label: torch.Tensor
) -> float:
    true = (prediction == label).float()
    total = prediction.shape[-1]**2
    return (torch.sum(true) / total).item()


def get_recall(
        prediction: torch.Tensor, 
        label: torch.Tensor
) -> float:
    tp = torch.sum((prediction == label).float())
    fp = torch.sum((prediction == (label + 1.)).float())
    return (tp / (tp + fp)).item()


def get_precision(
        prediction: torch.Tensor,
        label: torch.Tensor
) -> float:
    tp = torch.sum((prediction == label).float())
    fn = torch.sum((prediction == (label - 1)).float())
    return (tp / (tp + fn)).item()


def get_dice(
        prediction: torch.Tensor, 
        label: torch.Tensor
        ) -> float:
    numerator = torch.sum(prediction * label)
    denominator = torch.sum(prediction + label)
    return ((2*numerator) / denominator).item()


def get_iou(
        prediction: torch.Tensor, 
        label: torch.Tensor
        ) -> float:
    numerator = torch.sum(prediction * label)
    denominator = torch.sum(prediction + label) - numerator
    return (numerator / denominator).item()


def get_kernel_2d(
    l: int,
    sig: float
) -> torch.Tensor:
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)[None, None, :, :]
    kernel = kernel / np.sum(kernel)
    kernel = torch.from_numpy(kernel).float()
    return kernel


def get_margin_2d(
        image: torch.Tensor,
        l: int,
        sig: float
) -> torch.Tensor:
    blur_kernel = get_kernel_2d(l=l, sig=sig).to(image.device)
    x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float()[None, None, :, :].to(image.device)
    y_kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).float()[None, None, :, :].to(image.device)

    x_edge = torch.nn.functional.conv2d(image, weight=x_kernel)
    y_edge = torch.nn.functional.conv2d(image, weight=y_kernel)
    edge = torch.sqrt(x_edge.pow(2) + y_edge.pow(2))
    blurred = torch.nn.functional.conv2d(edge, blur_kernel)
    margin = (blurred > 0).float()
    return margin


def get_nsd_2d(
        prediction: torch.Tensor, 
        label: torch.Tensor, 
        l: int = 5, 
        sig: float = 5.
        ) -> float:
    prediction_margin = get_margin_2d(prediction, l=l, sig=sig)
    label_margin = get_margin_2d(label, l=l, sig=sig)
    numerator = torch.sum(prediction_margin * label_margin)
    denominator = torch.sum(prediction_margin + label_margin)
    return (2*numerator / denominator).item()

def floor_cap(
        number, 
        floor: float = 0., 
        cap: float = 1.
) -> float:
    if number <= floor:
        new_number = floor
    elif number >= cap:
        new_number = cap
    else: 
        new_number = number
    return new_number

class Divider:
    def __init__(self):
        self.count = 0.

    def add_sample(self, sample):
        self.count += (sample > 0).astype(np.uint8)

def non_zero_divider(new, factor):
    factor = np.array([factor if new[i] != 0 else 1 for i in range(new.shape[0])])
    return new / factor

def get_ece(prediction, label, other=None, n_bins=10):

    y_true = label.flatten()
    y_prob = prediction.flatten()
    y_true = y_true == 1
    device = y_prob.device

    bins = torch.linspace(1/n_bins, 1.0 - 1/n_bins, n_bins-1).to(device)
    binids = torch.searchsorted(bins[1:-1], y_prob).to(device)

    bin_sums = torch.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = torch.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = torch.bincount(binids, minlength=len(bins))

    prob_true = torch.nan_to_num(bin_true / bin_total)
    prob_pred = torch.nan_to_num(bin_sums / bin_total)
    return prob_true.cpu().numpy()[:-1]

def get_hist(stack, n_bins=100):
    mini, maxi = torch.min(stack), torch.max(stack)
    counts = torch.histc(stack, bins=n_bins, min=mini, max=maxi)
    boundaries = torch.linspace(mini, maxi, n_bins + 1)
    return (counts.cpu().numpy(), boundaries.cpu().numpy())

def get_auroc(prediction, label):
    predictions = prediction.flatten()
    labels = label.flatten()
    sorted_indices = torch.argsort(predictions, descending=True)
    sorted_predictions = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    tps = torch.cumsum(sorted_labels, dim=0)
    fps = torch.cumsum(1 - sorted_labels, dim=0)
    
    tps = torch.cat([torch.tensor([0.0], device=predictions.device), tps])
    fps = torch.cat([torch.tensor([0.0], device=predictions.device), fps])
    
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    
    auroc = torch.trapz(tpr, fpr).item()
    
    return auroc

def get_yarins(
        binary_prediction: torch.Tensor, 
        continuous_section: torch.Tensor,
        label: torch.Tensor, 
        section_threshold: float,
) -> Dict[str, float]:

    # section_certain, section_uncertain = threshold_above_below(continuous_section, section_threshold)
    section_uncertain, section_certain = threshold_above_below(continuous_section, section_threshold)
    prediction_accurate, prediction_innacurate = get_binary_accurate_inaccurate(binary_prediction, label)

    nac = torch.sum(prediction_accurate * section_certain)
    nau = torch.sum(prediction_accurate * section_uncertain)
    nic = torch.sum(prediction_innacurate * section_certain)
    niu = torch.sum(prediction_innacurate * section_uncertain)

    pa = torch.sum(prediction_accurate) / (torch.sum(prediction_accurate + prediction_innacurate))
    pi = 1 - pa

    pu = torch.sum(section_uncertain) / torch.sum(section_certain + section_uncertain)

    pac = nac / (nac + nic)
    pui = niu / (nic + niu)
    piu = pui * pi / pu
    pavpu = (nac + niu) / (nac + nau + nic + niu)
    out = np.array([pac.item(), pui.item(), piu.item(), pavpu.item()])
    return np.nan_to_num(out, nan=0.0)