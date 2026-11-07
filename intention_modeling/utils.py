import yaml
import os
import torch
from types import SimpleNamespace
from torchrl.envs import Transform
from tensordict.nn import TensorDictModule


MODEL_DIR = "saved_models"


def load_config(file_path: str) -> SimpleNamespace:
    with open(file_path, "r") as file:
        config_dict = yaml.full_load(file)

    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d

    return dict_to_namespace(config_dict)


def set_nested_attr(ns: SimpleNamespace, attr_path: str, value):
    parts = attr_path.split(".")
    curr = ns
    for p in parts[:-1]:
        curr = getattr(curr, p)
    setattr(curr, parts[-1], value)


def apply_overrides(config: SimpleNamespace, override_str: str):
    if not override_str:
        return

    overrides = override_str.split(",")
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Invalid override format: {ov}")

        path, val = ov.split("=", 1)

        try:
            parsed_val = yaml.safe_load(val)
        except yaml.YAMLError:
            parsed_val = val

        set_nested_attr(config, path, parsed_val)


class DoneTransform(Transform):

    def __init__(self, env):
        super().__init__()
        self.groups = env.group_map.keys()

    def forward(self, tensordict):
        for group in self.groups:
            keys = list(tensordict.keys(True, True))
            group_shape = tensordict.get_item_shape(group)
            nested_done_key = ("next", group, "done")
            nested_terminated_key = ("next", group, "terminated")

            if nested_done_key not in keys:
                tensordict.set(
                    nested_done_key,
                    tensordict.get(("next", "done"))
                    .unsqueeze(-1)
                    .expand((*group_shape, 1)),
                )
            if nested_terminated_key not in keys:
                tensordict.set(
                    nested_terminated_key,
                    tensordict.get(("next", "terminated"))
                    .unsqueeze(-1)
                    .expand((*group_shape, 1)),
                )
        return tensordict


def save_model(obj, save_name):
    folder_path = os.path.join(MODEL_DIR, save_name)
    os.makedirs(folder_path, exist_ok=True)

    existing_files = [
        int(f.split(".")[0]) for f in os.listdir(folder_path) if f.endswith(".pth")
    ]

    save_path = os.path.join(folder_path, f"{len(existing_files)}.pth")
    torch.save(obj, save_path)


def save_checkpoint(save_name, intention_model, embedding_mixer, main_agent, meta):
    folder_path = os.path.join(MODEL_DIR, save_name)
    os.makedirs(folder_path, exist_ok=True)

    encoders = {}
    decoders = {}
    reward_projs = {}
    for im in intention_model.encoder_decoder_list:
        encoders[im.model] = im.encoder.state_dict()
        if hasattr(im, "decoders"):
            decoders[im.model] = {
                name: dec.state_dict() for name, dec in im.decoders.items()
            }
        elif hasattr(im, "decoder"):
            decoders[im.model] = im.decoder.state_dict()
        if hasattr(im, "reward_proj"):
            reward_projs[im.model] = im.reward_proj.state_dict()

    mixer_module = None if embedding_mixer is None else embedding_mixer.module
    checkpoint = {
        "encoders": encoders,
        "decoders": decoders,
        "reward_projs": reward_projs,
        "mixer": None if mixer_module is None else mixer_module.state_dict(),
        "expert_names": None
        if mixer_module is None
        else getattr(mixer_module, "expert_names", None),
        "policy": main_agent.policy.state_dict(),
        "meta": meta,
    }
    torch.save(checkpoint, os.path.join(folder_path, "checkpoint.pth"))


def load_all_models(save_name, device):
    folder_path = os.path.join(MODEL_DIR, save_name)
    if not os.path.exists(folder_path) or not os.listdir(folder_path):
        raise FileNotFoundError(f"No saved models found in {folder_path}")

    model_files = [f for f in os.listdir(folder_path) if f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError(f"No valid model files found in {folder_path}")

    return [
        torch.load(
            os.path.join(folder_path, file), map_location=device, weights_only=False
        )
        for file in model_files
    ]
