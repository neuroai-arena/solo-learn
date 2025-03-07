from typing import List, Dict, Any, Union, Optional, Generator, Tuple
from itertools import product
import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(
            self,
            out_dim: int,
            num_classes: int,
            **output_kwargs,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.output_kwargs = output_kwargs

        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        ret, output = create_linear_input(x_tokens_list, **self.output_kwargs)

        if not ret:
            raise RuntimeError(f"Linear layer returned no output.")
        if output.shape[-1] != self.out_dim:
            raise ValueError(
                f"Output shape {output.shape[-1]} does not match the expected input dimension {self.out_dim} of clf {self.use_avgpool=}, {self.use_n_blocks=}"
            )
        return self.linear(output)


class LinearClassifierSimple(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim: int, num_classes: int = 1000):
        super().__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes

        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        return self.linear(x)


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


def scale_lr(learning_rates: float, batch_size: int, devices: int) -> float:
    return learning_rates * (batch_size * devices) / 256.0


def setup_linear_classifiers_only_lr(
        sample_output: torch.Tensor,
        learning_rates: List[float],
        batch_size: int,
        devices: int,
        num_classes: int,
) -> Union[AllClassifiers, List[Dict[str, Any]]]:
    print("Setting up linear classifiers:")
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for _lr in learning_rates:

        lr = scale_lr(_lr, batch_size, devices)

        out_dim = sample_output.shape[-1]

        linear_classifier = LinearClassifierSimple(out_dim, num_classes=num_classes)
        clf_str = f"classifier-lr_{lr:.8f}".replace(".", ":")
        print("\t- ", clf_str)
        linear_classifiers_dict[clf_str] = linear_classifier

        optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)

    return linear_classifiers, optim_param_groups


def parameter_iterator(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    for k, v in params.items():
        if v is None:
            params.pop(k)

    keys = params.keys()
    values = params.values()

    return [dict(zip(keys, combination)) for combination in product(*values)]


def setup_linear_classifiers_transformer(
        batch_size: int,
        devices: int,
        num_classes: int,
        param_dict: Dict[str, List[Any]],
        sample_output: torch.Tensor,
        has_class_token: bool,

) -> Union[AllClassifiers, List[Dict[str, Any]]]:
    print("Setting up linear classifiers:")

    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []

    if not 'lr' in param_dict:
        raise ValueError("Learning rate must be specified in the parameter dictionary.")

    for param in parameter_iterator(param_dict):
        lr = param.pop('lr')
        lr = scale_lr(lr, batch_size, devices)

        ret, out_dim = create_linear_input(sample_output, has_class_token=has_class_token, **param)
        if not ret:
            continue

        out_dim = out_dim.shape[-1]

        linear_classifier = LinearClassifier(out_dim, num_classes=num_classes, has_class_token=has_class_token, **param)

        clf_str ="classifier-" +  "-".join([f'{key}_{value}' for key, value in param.items()]) + f"-lr_{lr:.8f}"
        print("\t- ", clf_str)
        linear_classifiers_dict[clf_str.replace(".", ":")] = linear_classifier

        optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)

    return linear_classifiers, optim_param_groups


def setup_linear_classifiers(
        sample_output: torch.Tensor,
        learning_rates: List[float],
        has_class_token: bool,
        batch_size: int,
        devices: int,
        num_classes: int,
        is_transformer: bool,
        # transformer search params
        use_avgpool: List[bool] = None,
        use_cls_token: List[bool] = None,
        use_n_blocks: List[int] = None,
) -> Union[AllClassifiers, List[Dict[str, Any]]]:
    special_search = any([x is not None for x in [use_avgpool, use_cls_token, use_n_blocks]])

    if is_transformer and special_search:
        return setup_linear_classifiers_transformer(
            sample_output=sample_output,
            has_class_token=has_class_token,
            batch_size=batch_size,
            devices=devices, num_classes=num_classes,
            param_dict=dict(use_avgpool=use_avgpool, use_cls_token=use_cls_token, use_n_blocks=use_n_blocks, lr=learning_rates),
        )
    else:
        return setup_linear_classifiers_only_lr(
            sample_output, learning_rates, batch_size, devices, num_classes
        )


def create_linear_input(
        x_tokens: torch.Tensor,
        use_n_blocks: int,
        use_avgpool: bool,
        use_cls_token: bool,
        has_class_token: bool = True,
) -> Tuple[bool, torch.Tensor]:
    """
    Create the input for the linear head from the output of the Vision Transformer.

    It could be either the concatenation of the class token of the last `use_n_blocks` blocks or
    the concatenation of the class token of the last `use_n_blocks` blocks and the average pooled
    version of the last block's patch tokens.

    Args:
        x_tokens: The output of the Vision Transformer. Shape (batch, layer, tokens, dim).
        use_n_blocks: The number of blocks to use.
        use_avgpool: Whether to use average pooling or not.
        has_class_token: Whether the transformer has a class token or not.
    """
    if not has_class_token and not use_avgpool:
        print("Cannot use `use_avgpool=False` when the transformer has no cls tokens.")
        return False, None
    if use_cls_token and not has_class_token:
        print("Cannot use `use_cls_token=True` when the transformer has no cls tokens.")
        return False, None
    if not use_cls_token and not use_avgpool:
        print("Cannot use both `use_cls_token=False` and `use_avgpool=False`.")
        return False, None

    # use the last `use_n_blocks` blocks
    intermediate_output = x_tokens[:, -use_n_blocks:, :, :]  # (batch, n_last_blocks, tokens, dim)

    if has_class_token:
        if use_cls_token:
            # take the class token of the last `use_n_blocks` blocks
            cls = intermediate_output[:, :, 0, :].reshape(intermediate_output.shape[0],
                                                          -1)  # (batch, n_last_blocks * dim)

        if use_avgpool:
            # average pool the patch tokens of the last block
            avgpool = intermediate_output[:, :, 1, :].reshape(intermediate_output.shape[0],
                                                                                  -1)  # (batch, n_last_blocks * dim)

        if use_avgpool and use_cls_token:
            output = torch.cat((cls, avgpool), dim=-1)
            output = output.reshape(output.shape[0], -1)
        elif use_avgpool:
            output = avgpool
        elif use_cls_token:
            output = cls
        else:
            raise ValueError("Cannot have both `use_cls_token=False` and `use_avgpool=False`.")
    else:
        # take the average pool of the last `use_n_blocks` blocks and concatenate them
        output = torch.mean(intermediate_output, dim=2).reshape(intermediate_output.shape[0], -1)
    return True, output.contiguous()
