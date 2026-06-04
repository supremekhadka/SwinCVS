print("Importing libraries...")
# Standard library imports
import argparse
import time
import json
from pathlib import Path
import warnings
import os

# Third-party imports
import torch
import numpy as np

# Local imports
from scripts.f_environment import (
    get_config,
    set_deterministic_behaviour,
    verify_results_weights_folder,
)
from scripts.f_dataset import get_inference_dataset, get_inference_dataloader
from scripts.f_build import build_inference_model
from scripts.f_metrics import get_map, get_balanced_accuracies


warnings.filterwarnings("ignore")

##############################################################################################
##############################################################################################
# ENVIRONMENT
pwd = Path.cwd()
print(f"Current working directory: {pwd}")

# Verify necessary folder structure and download weights
verify_results_weights_folder(pwd, mode="inference")

# Load config
parser = argparse.ArgumentParser(
    description="Run Inference on SwinCVS with specified config"
)
parser.add_argument(
    "--inference_config_path",
    type=str,
    required=True,
    help="Path to inference config YAML file",
)
args = parser.parse_args()
config_path = args.config_path
config = get_config(config_path, mode="inference")

seed = config.SEED
set_deterministic_behaviour(seed)


##############################################################################################
##############################################################################################
# DATASET and DATALOADER
dataset = get_inference_dataset(config)
dataloader = get_inference_dataloader(config, dataset)

##############################################################################################
##############################################################################################

# Initialise SwinCVS according to config
model = None
model = build_inference_model(config)
print("Full model initialised successfully!\n")

# Load saved weights for inference
weights = "weights/" + config.WEIGHTS
model.load_state_dict(torch.load(weights))
print(
    f"Trained SwinCVS weights loaded successfully for INFERENCE - name: {config.MODEL.INFERENCE_WEIGHTS}"
)
if config.CUDA_ID:
    model.to("cuda:{CUDA_ID}")
else:
    model.to("cuda")

torch.cuda.empty_cache()

results_dict = {}

# Test time measurement variables
start_time = 0
end_time = 0
times = []

# Performance measurement variables
test_probabilities = []
test_predictions = []
test_targets = []

len_dataloader = len(dataloader)
print("\nTesting")
model.eval()
with torch.inference_mode():
    for idx, (samples, targets) in enumerate(dataloader):
        print(f"Processing batch: {idx + 1:04}/{len_dataloader:04}", end="\r")

        # Time start
        start_time = time.time()

        # Get preds
        samples, targets = samples.to("cuda"), targets.to("cuda")

        outputs_lstm = model(samples)

        # Get outputs
        test_probability = torch.sigmoid(outputs_lstm)
        test_prediction = torch.round(test_probability)

        # Time end
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

        # Save results from a batch to a list
        test_probabilities.append(test_probability.to("cpu"))
        test_predictions.append(test_prediction.to("cpu"))
        test_targets.append(targets.to("cpu"))

        torch.cuda.synchronize()

# Calculate metrics
(
    C1_balanced_accuracy,
    C2_balanced_accuracy,
    C3_balanced_accuracy,
    total_balanced_accuracy,
) = get_balanced_accuracies(test_targets, test_predictions)
C1_ap, C2_ap, C3_ap, mAP = get_map(test_targets, test_probabilities)

# Print metrics
print("\nTesting results:")
print(
    "Average balanced accuracy",
    round((C1_balanced_accuracy + C1_balanced_accuracy + C3_balanced_accuracy) / 3, 4),
)
print("C1 bacc", round(C1_balanced_accuracy, 4))
print("C2 bacc", round(C2_balanced_accuracy, 4))
print("C3 bacc", round(C3_balanced_accuracy, 4))
print("mAP", round(mAP, 4))
print("C1 ap", round(C1_ap, 4))
print("C2 ap", round(C2_ap, 4))
print("C3 ap", round(C3_ap, 4))

mean_inference = round(np.mean(times) * 1000, 1)
std_inference = round(np.std(times) * 1000, 1)
total_inference = round(np.sum(times), 1)
print(
    f"Inference time: mean={mean_inference}ms, std={std_inference}ms, total={total_inference}s"
)

test_predictions_2save = torch.cat(test_predictions, dim=0).tolist()
test_probabilities_2save = torch.cat(test_probabilities, dim=0).tolist()
test_targets_2save = torch.cat(test_targets, dim=0).tolist()

inference_results = {
    "avg_bal_acc": round(total_balanced_accuracy, 4),
    "C1_bacc": round(C1_balanced_accuracy, 4),
    "C2_bacc": round(C2_balanced_accuracy, 4),
    "C3_bacc": round(C3_balanced_accuracy, 4),
    "avg_map": round(mAP, 4),
    "C1_map": round(C1_ap, 4),
    "C2_map": round(C2_ap, 4),
    "C3_map": round(C3_ap, 4),
    "preds": test_predictions_2save,
    "true": test_targets_2save,
    "preds_prob": test_probabilities_2save,
    "mean_inference_ms": mean_inference,
    "std_inference_ms": std_inference,
    "total_inference_s": total_inference,
}
results_dict["Inference_Results"] = inference_results

os.makedirs(pwd / "results" / "inference", exist_ok=True)

with open(
    pwd / "results" / "inference" / f"{config.INFERENCE_NAME}results.json", "w"
) as file:
    json.dump(results_dict, file, indent=4)
