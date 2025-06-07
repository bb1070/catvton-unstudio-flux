from huggingface_hub import upload_folder, login
import os

model_dir = "./trained-flux-inpaint/checkpoint-500"  # this points to the folder on your local machine where the trained model is stored

repo_id = "bb1070/catvton-flux-acne-2000" # change this to your actual repo id on huggingface to upload your checkpoints or model

# Upload the entire folder
upload_folder(
    repo_id=repo_id,
    folder_path=model_dir,
    repo_type="model",
    allow_patterns=["*.bin", "*.json", "*.safetensors", "*.txt", "*.model", "*.pt", "*.yaml", "*config*", "*tokenizer*"],
    commit_message="Uploading full fine-tuned FLUX model"
)
