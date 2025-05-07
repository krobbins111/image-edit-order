import os
import json
from typing import List, Tuple
from openai import OpenAI
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from pydantic import BaseModel

class AtomicEdits(BaseModel):
    edit_list: List[str]
    edit_levels: List[int]

client = OpenAI(api_key="")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Img2Img
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(DEVICE)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")


def generate_atomic_edits(description, model="gpt-4o"):
    prompt = (
        "You are an assistant that extracts a list of atomic img2img text prompts from a description. Each prompt must have an edit with a place. Each edit should start with 'a hotel room with...'. Each edit should also come with a score from 1 to 3 depending on how much of the image needs to visually change."
        "Given the description, output a JSON array of concise atomic descriptions in the form of img2img prompts. turning each part of the description into individual atomic description that can be made with prompts to an img2img model.\n\n"
        f"Description: {description}\n"
        "Edits (JSON array):"
    )
    resp = client.responses.parse(
        model=model,
        input=prompt,
        text_format=AtomicEdits,
    )
    print(resp.output_parsed.edit_list)
    return resp.output_parsed.edit_list, resp.output_parsed.edit_levels

def embed_text(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
    return emb / emb.norm(dim=-1, keepdim=True)

def embed_image(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb / emb.norm(dim=-1, keepdim=True)

def apply_edit(input_image, edit_instruction, strength=0.8, guidance_scale= 1.75):
    edit_description, level = edit_instruction
    print(level, (level * 0.1) + 0.65)
    out = pipe(
        prompt=edit_description,
        image=input_image,
        strength=(level * 0.1) + 0.65,
        guidance_scale=guidance_scale,
        num_inference_steps=4
    )
    return out.images[0]




# Backtracking
best_score = -float("inf")
best_sequence = []
best_image = None


def backtrack(current_image, current_embedding, completed, edit_embeddings, edits, prune_drop=True):
    global best_score, best_sequence, best_image

    if completed:
        sims = [torch.cosine_similarity(current_embedding, edit_embeddings[edits.index(e)]).item() for e in completed]
        prev_score = sum(sims) / len(sims)
    else:
        prev_score = 0.0

    if len(completed) == len(edits):
        if prev_score > best_score:
            best_score, best_sequence, best_image = prev_score, completed.copy(), current_image
        return

    remaining = [e for e in edits if e not in completed]
    remaining.sort(key=lambda e: e[1], reverse=True)
    for edit in remaining:
        new_image = apply_edit(current_image, edit)
        new_embedding = embed_image(new_image)
        similarity = torch.cosine_similarity(new_embedding, edit_embeddings[edits.index(edit)]).item()
        new_score = (prev_score * len(completed) + similarity) / (len(completed) + 1)
        if prune_drop and similarity < prev_score - 0.01:
            continue

        completed.append(edit)
        backtrack(new_image, new_embedding, completed, edit_embeddings, edits, prune_drop)
        completed.pop()


def display_sequence(base_image, sequence):
    sequence_images = [("Base Image", base_image)]
    current = base_image
    for edit in sequence:
        current = apply_edit(current, edit)
        sequence_images.append((edit, current))

    n = len(sequence_images)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    for ax, (label, img) in zip(axes, sequence_images):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    base_image = Image.open("hotel.png").convert("RGB")
    description = (
        "Floor to ceiling windows showing the city skyscrapers. Red sheets with gold patterns on the design. A small white recliner in the back corner of the room. There was a large cicular light on the ceiling."
    )
    edits, edit_levels = generate_atomic_edits(description)

    edit_embeddings = [embed_text(e) for e in edits]

    image_embedding = embed_image(base_image)

    backtrack(base_image, image_embedding, [], edit_embeddings, list(zip(edits, edit_levels)), prune_drop=True)

    print("Best average similarity:", best_score)
    print("Best edit sequence:", best_sequence)
    best_image.save("best_edited_room.png")

    # Display the sequence of images
    display_sequence(base_image, best_sequence)

if __name__ == "__main__":
    main()
