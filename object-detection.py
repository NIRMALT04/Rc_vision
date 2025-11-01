from google import genai
from google.genai import types
from PIL import Image, ImageDraw
import io
import base64
import json
import numpy as np
import os

client = genai.Client(api_key="YOUR_KEY" )

def parse_json(json_output: str):
  # Parsing out the markdown fencing
  lines = json_output.splitlines()
  output = json_output  # Default to original if no markdown found
  
  for i, line in enumerate(lines):
    if line.strip() == "```json":
      # Get content after ```json
      content_after = "\n".join(lines[i+1:])
      # Split by closing ``` to get just the JSON
      output = content_after.split("```")[0].strip()
      break  # Exit the loop once "```json" is found
  
  # Try to find JSON array start if not already found
  if output and not output.strip().startswith("["):
    # Try to find JSON array in the text
    start_idx = output.find("[")
    if start_idx != -1:
      # Find matching closing bracket by properly handling strings
      bracket_count = 0
      in_string = False
      escape_next = False
      
      for i in range(start_idx, len(output)):
        char = output[i]
        
        if escape_next:
          escape_next = False
          continue
        
        if char == '\\':
          escape_next = True
          continue
        
        if char == '"':
          in_string = not in_string
          continue
        
        if not in_string:
          if char == "[":
            bracket_count += 1
          elif char == "]":
            bracket_count -= 1
            if bracket_count == 0:
              output = output[start_idx:i+1]
              break
  
  return output

def extract_segmentation_masks(image_path: str, output_dir: str = "segmentation_outputs"):
  # Load and resize image
  im = Image.open(image_path)
  im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

  prompt = """
  Analyze this image and provide segmentation masks for ALL objects visible in the image.
  Detect and segment all items regardless of their material type (wood, glass, plastic, metal, paper, fabric, electronic, etc.).
  Output a JSON list of segmentation masks where each entry contains:
  - The 2D bounding box in the key "box_2d" (as an array of 4 values: [y0, x0, y1, x1] in normalized coordinates 0-1000)
  - The segmentation mask in key "mask" (as a base64-encoded PNG image)
  - The text label in the key "label" (descriptive label including material type if identifiable) 
  - 
  
  Use descriptive labels that include the object name and material when possible.
  For example: "plastic water bottle", "wooden table", "glass container", "metal can", etc.
  
  Segment all visible objects, not just specific material types.
  """

  config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=0) # set thinking_budget to 0 for better results in object detection
  )

  response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[prompt, im], # Pillow images can be directly passed as inputs (which will be converted by the SDK)
    config=config
  )

  # Parse JSON response
  parsed_text = parse_json(response.text)
  try:
    items = json.loads(parsed_text)
  except json.JSONDecodeError as e:
    print(f"JSON parsing error: {e}")
    print(f"Parsed text (first 500 chars): {parsed_text[:500]}")
    print(f"Parsed text (last 500 chars): {parsed_text[-500:]}")
    # Return empty result if no items detected
    os.makedirs(output_dir, exist_ok=True)
    return None

  # Create output directory
  os.makedirs(output_dir, exist_ok=True)

  # Create a single combined overlay for all masks
  combined_overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
  combined_draw = ImageDraw.Draw(combined_overlay)

  # Process each mask
  for i, item in enumerate(items):
      # Get bounding box coordinates
      box = item["box_2d"]
      y0 = int(box[0] / 1000 * im.size[1])
      x0 = int(box[1] / 1000 * im.size[0])
      y1 = int(box[2] / 1000 * im.size[1])
      x1 = int(box[3] / 1000 * im.size[0])

      # Skip invalid boxes
      if y0 >= y1 or x0 >= x1:
          continue

      # Process mask
      png_str = item["mask"]
      if not png_str.startswith("data:image/png;base64,"):
          continue

      # Remove prefix
      png_str = png_str.removeprefix("data:image/png;base64,")
      mask_data = base64.b64decode(png_str)
      mask = Image.open(io.BytesIO(mask_data))

      # Resize mask to match bounding box
      mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)

      # Convert mask to numpy array for processing
      mask_array = np.array(mask)

      # Add mask to combined overlay with different colors for different objects
      colors = [
          (255, 0, 0, 200),    # Red
          (0, 255, 0, 200),    # Green
          (0, 0, 255, 200),    # Blue
          (255, 255, 0, 200),  # Yellow
          (255, 0, 255, 200),  # Magenta
          (0, 255, 255, 200),  # Cyan
      ]
      color = colors[i % len(colors)]
      
      for y in range(y0, y1):
          for x in range(x0, x1):
              if y - y0 < mask_array.shape[0] and x - x0 < mask_array.shape[1]:
                  # Handle both grayscale (2D) and RGB (3D) masks
                  if len(mask_array.shape) == 2:
                      pixel_value = mask_array[y - y0, x - x0]
                  else:
                      pixel_value = mask_array[y - y0, x - x0, 0]
                  
                  if pixel_value > 128:  # Threshold for mask
                      combined_draw.point((x, y), fill=color)

      # Save individual mask for reference
      mask_filename = f"{item['label']}_{i}_mask.png"
      mask.save(os.path.join(output_dir, mask_filename))
      print(f"Saved mask for {item['label']} to {output_dir}")

  # Create and save single combined overlay image
  if len(items) > 0:
      composite = Image.alpha_composite(im.convert('RGBA'), combined_overlay)
      combined_filename = os.path.join(output_dir, "combined_overlay.png")
      composite.save(combined_filename)
      print(f"Saved combined overlay with {len(items)} masks to {output_dir}")
      return combined_filename
  else:
      return None

# Example usage
if __name__ == "__main__":
  extract_segmentation_masks("frame.jpg")
