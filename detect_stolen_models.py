import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from safetensors.torch import load_file

# CONFIG

DEVICE = "cpu"

TARGET_MODEL_PATH = "target_model/weights.safetensors"
SUSPECT_MODELS_DIR = "suspect_models"

DATA_ROOT = "./cifar100"

SUBSET_SIZE = 1024
BATCH_SIZE = 128

# MULTI-NOISE ROBUSTNESS
NOISE_LEVELS = [0.02, 0.05, 0.10]

OUTPUT_CSV = "submission.csv"


# MODEL
def make_model():

    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()

    model.fc = nn.Linear(
        model.fc.in_features,
        100,
    )

    return model


def load_model(path):

    model = make_model()

    state_dict = load_file(
        path,
        device="cpu",
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model

# DATA
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    )
])

# HARD SAMPLE SELECTION
def get_loader():

    dataset = datasets.CIFAR100(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform,
    )

    temp_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("Selecting hardest samples...")

    target_model = load_model(
        TARGET_MODEL_PATH
    )

    entropies = []

    with torch.no_grad():

        for images, _ in tqdm(temp_loader):

            logits = target_model(images)

            probs = F.softmax(
                logits,
                dim=1,
            )

            entropy = -(
                probs * torch.log(probs + 1e-8)
            ).sum(dim=1)

            entropies.extend(
                entropy.cpu().numpy()
            )

    entropies = np.array(entropies)

    # highest entropy samples
    hard_indices = np.argsort(
        entropies
    )[-SUBSET_SIZE:]

    subset = Subset(
        dataset,
        hard_indices.tolist(),
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return loader

# PREDICTIONS
def collect_predictions(model, loader):

    preds = []

    with torch.no_grad():

        for images, _ in loader:

            logits = model(images)

            pred = logits.argmax(dim=1)

            preds.append(pred)

    return torch.cat(preds)

# MULTI-NOISE ROBUSTNESS
def noisy_stability(model, loader):

    stabilities = []

    with torch.no_grad():

        for noise_std in NOISE_LEVELS:

            clean_preds_all = []
            noisy_preds_all = []

            for images, _ in loader:

                clean_logits = model(images)

                noise = (
                    torch.randn_like(images)
                    * noise_std
                )

                noisy_images = images + noise

                noisy_logits = model(
                    noisy_images
                )

                clean_preds = clean_logits.argmax(dim=1)

                noisy_preds = noisy_logits.argmax(dim=1)

                clean_preds_all.append(
                    clean_preds
                )

                noisy_preds_all.append(
                    noisy_preds
                )

            clean_preds_all = torch.cat(
                clean_preds_all
            )

            noisy_preds_all = torch.cat(
                noisy_preds_all
            )

            stability = (
                clean_preds_all
                == noisy_preds_all
            ).float().mean().item()

            stabilities.append(
                stability
            )

    return float(np.mean(stabilities))

# AGREEMENT

def prediction_agreement(a, b):

    return (
        a == b
    ).float().mean().item()

# CONFIDENCE SIGNAL
def confidence_similarity(
    model_a,
    model_b,
    loader,
):

    conf_a = []
    conf_b = []

    with torch.no_grad():

        for images, _ in loader:

            logits_a = model_a(images)
            logits_b = model_b(images)

            probs_a = F.softmax(
                logits_a,
                dim=1,
            )

            probs_b = F.softmax(
                logits_b,
                dim=1,
            )

            conf_a.append(
                probs_a.max(dim=1).values
            )

            conf_b.append(
                probs_b.max(dim=1).values
            )

    conf_a = torch.cat(conf_a)
    conf_b = torch.cat(conf_b)

    return F.cosine_similarity(
        conf_a.unsqueeze(0),
        conf_b.unsqueeze(0),
    ).item()

# NORMALIZATION
def normalize(values):

    values = np.array(values)

    return (
        (values - values.min())
        / (values.max() - values.min() + 1e-8)
    )

# MAIN
def main():

    print("Preparing dataloader...")

    loader = get_loader()

    print("Loading target model...")

    target_model = load_model(
        TARGET_MODEL_PATH
    )

    print("Collecting target predictions...")

    target_preds = collect_predictions(
        target_model,
        loader,
    )

    print("Computing target robustness...")

    target_stability = noisy_stability(
        target_model,
        loader,
    )

    rows = []

    print("Processing suspect models...")

    for model_id in tqdm(range(360)):

        checkpoint_path = os.path.join(
            SUSPECT_MODELS_DIR,
            f"suspect_{model_id:03d}.safetensors",
        )

        if not os.path.exists(
            checkpoint_path
        ):
            continue

        suspect_model = load_model(
            checkpoint_path
        )

        suspect_preds = collect_predictions(
            suspect_model,
            loader,
        )

        # AGREEMENT
        agreement = prediction_agreement(
            target_preds,
            suspect_preds,
        )

        # ROBUSTNESS GAP
        suspect_stability = noisy_stability(
            suspect_model,
            loader,
        )

        robustness_gap = 1.0 - abs(
            target_stability
            - suspect_stability
        )

        # CONFIDENCE MATCH
        conf_sim = confidence_similarity(
            target_model,
            suspect_model,
            loader,
        )

        # FINAL SCORE
        score = (
            0.55 * agreement +
            0.30 * robustness_gap +
            0.15 * conf_sim
        )

        rows.append({
            "id": model_id,
            "score": score,
        })

        del suspect_model

    # SAVE
    df = pd.DataFrame(rows)

    df["score"] = normalize(
        df["score"]
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\nSaved:", OUTPUT_CSV)

    print(
        df.sort_values(
            "score",
            ascending=False,
        ).head(10)
    )

# ENTRY
if __name__ == "__main__":
    main()
