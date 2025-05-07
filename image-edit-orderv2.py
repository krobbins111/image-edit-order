import os
import json
from typing import List, Tuple
from openai import OpenAI
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel, SamProcessor, SamModel, AutoProcessor, AutoModelForZeroShotObjectDetection
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler, ControlNetModel, AutoPipelineForInpainting, StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from pydantic import BaseModel
from google.colab.patches import cv2_imshow
import cv2 as cv

class AtomicEdits(BaseModel):
  edit_list: List[str]
  edit_target: List[str]

client = OpenAI(api_key="")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# GroundingDino
GD_MODEL_ID = "IDEA-Research/grounding-dino-base"
gd_processor = AutoProcessor.from_pretrained(GD_MODEL_ID)
gd_model     = AutoModelForZeroShotObjectDetection.from_pretrained(GD_MODEL_ID).to(DEVICE)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to(DEVICE)


def generate_atomic_edits(description, model="gpt-4o"):
  prompt = (
      "You are an assistant that extracts a list of atomic img2img text prompts from a description. Each prompt must have an edit and a target location/object to edit. For example and edit of 'the walls were blue wallpaper with planets on it' would have edit: 'blue wallpaper with planets on it' and location: 'walls'. Another example would be 'the lamp had a round paper shade' with edit: 'round paper shade' and location: 'lamp'"
      "Given the description, output a JSON array of concise atomic descriptions in the form of img2img prompts. turning each part of the description into individual atomic description that can be made with prompts to an img2img model.\n\n"
      f"Description: {description}\n"
      "Edits (JSON array):"
  )
  resp = client.responses.parse(
      model=model,
      input=prompt,
      text_format=AtomicEdits,
  )
  print(resp.output_parsed.edit_list, resp.output_parsed.edit_target)
  return resp.output_parsed.edit_list, resp.output_parsed.edit_target

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

def generate_box(image, edit_target, box_threshold=0.25, text_threshold=0.1):
  inputs = gd_processor(images=image, text=edit_target, return_tensors="pt").to(DEVICE)
  with torch.no_grad():
      outputs = gd_model(**inputs)
  results = gd_processor.post_process_grounded_object_detection(
      outputs, inputs.input_ids,
      box_threshold=box_threshold,
      text_threshold=text_threshold,
      target_sizes=[image.size[::-1]]
  )[0]
  boxes = results["boxes"]
  scores = results["scores"]
  if len(boxes) == 0:
      print(f"No bounding box found for “{edit_target}”")
      return None
  best_idx = torch.argmax(scores).item()
  x0, y0, x1, y1 = map(int, boxes[best_idx].tolist())
  width, height = x1 - x0, y1 - y0
  # dx = int(width * 0.1)
  # dy = int(height * 0.1)
  mask = np.zeros((image.height, image.width), dtype=np.uint8)
  # x0, y0 = max(0, x0 - dx), max(0, y0 - dy)
  # x1, y1 = min(image.width, x1 + dx), min(image.height, y1 + dy)
  mask[y0:y1, x0:x1] = 255

  mask_image = Image.fromarray(mask)
  mask_image = mask_image.resize(image.size, resample=Image.NEAREST)
  cv2_imshow(cv.cvtColor(np.array(mask_image), cv.COLOR_RGB2BGR))
  return mask_image

def apply_edit(input_image, edit_instruction, strength=0.85, guidance_scale=14):
  edit_description, edit_target = edit_instruction
  mask = generate_box(input_image, edit_target)
  if mask is None:
      return None
  out = pipe(
      mask_image=mask,
      prompt=edit_description,
      image=input_image,
      strength=strength,
      guidance_scale=guidance_scale,
      # num_inference_steps=25
  )
  cv2_imshow(cv.cvtColor(np.array(out.images[0]), cv.COLOR_RGB2BGR))
  return out.images[0]




# Backtracking
best_score = -float("inf")
best_sequence = []
best_image = None


def backtrack(current_image, current_embedding, completed, edit_embeddings, edits, prune_drop=True):
  global best_score, best_sequence

  if completed:
    sims = [torch.cosine_similarity(current_embedding, edit_embeddings[edits.index(e)]).item() for e in completed]
    prev_score = sum(sims) / len(sims)
  else:
    prev_score = 0.0

  if len(completed) == len(edits):
    if prev_score > best_score:
      best_score, best_sequence = prev_score, completed.copy()
      best_image = current_image.copy()
    return current_image

  remaining = [e for e in edits if e not in completed]
  # remaining.sort(key=lambda e: e[1], reverse=True)
  for edit in remaining:
    print(edit)
    new_image = apply_edit(current_image, edit)
    # This sequence lost the target in a previous edit perhaps
    if new_image is None:
      print('No image generated.')
      continue
    new_embedding = embed_image(new_image)
    similarity = torch.cosine_similarity(new_embedding, edit_embeddings[edits.index(edit)]).item()
    new_score = (prev_score * len(completed) + similarity) / (len(completed) + 1)
    # Require that the average score got better and the newest edit was acceptable
    if prune_drop and new_score < prev_score - 0.05:
      print("New image was much worse than previous on average.")
      continue
    if prune_drop and similarity < 0.15:
      print("New image was not a good representation of the prompt.")
      continue

    completed.append(edit)
    backtrack(new_image, new_embedding, completed, edit_embeddings, edits, prune_drop)
    completed.pop()
  return False


def display_sequence(base_image, sequence):
  sequence_images = [("Base Image", base_image)]
  current = base_image
  for edit in sequence:
      current = apply_edit(current, edit)
      sequence_images.append((edit[0], current))
  sequence_images[-1][1].save("best_edited_room.png")

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
  global best_score, best_sequence, best_image
  base_image = Image.open("hotel.png").convert("RGB")
  description = (
      "Floor to ceiling windows showing the city skyscrapers. Red sheets with gold patterns on the design. A small white recliner in the back corner of the room. There was a large cicular light on the ceiling."
  )
  edits = ["the hotel room's curtains were beige.", "the hotel room's sheets were red with gold patterns.", "a hotel room's small white recliner.", "a hotel room's large gold circular light fixture on the ceiling."]
  edit_target = ['the curtains.', 'on the bed.', 'a sofa chair in the corner.', 'the ceiling.']
  # edits, edit_target = generate_atomic_edits(description)
  # uncomment above line and comment 2 above to have gpt do this for you from the description

  edit_embeddings = [embed_text(e) for e in edits]

  image_embedding = embed_image(base_image)

  final_image = backtrack(base_image, image_embedding, [], edit_embeddings, list(zip(edits, edit_target)), prune_drop=True)

  print("Best average similarity:", best_score)
  print("Best edit sequence:", best_sequence)
  # if best_image != None:
  #   best_image.save("best_edited_room.png")
  # Display the sequence of images
  display_sequence(base_image, best_sequence)

  retrieval_target = Image.open("retrieve_me.jpg")
  original_image = Image.open("hotel.png")
  best_image = Image.open("best_edited_room.png")
  retrieval_embedding = embed_image(retrieval_target)
  original_embedding = embed_image(original_image)
  best_embedding = embed_image(best_image)
  similarity_retrieval = [torch.cosine_similarity(retrieval_embedding, eb).item() for eb in edit_embeddings]
  similarity_original = [torch.cosine_similarity(original_embedding, eb).item() for eb in edit_embeddings]
  similarity_best = [torch.cosine_similarity(best_embedding, eb).item() for eb in edit_embeddings]
  print(f"Similarity average for 'hotel.jpg': {similarity_original}")
  print(f"Similarity average for 'best_edited_room.png': {similarity_best}")
  print(f"Similarity average for 'retrieve_me.jpg': {similarity_retrieval}")

if __name__ == "__main__":
    main()
