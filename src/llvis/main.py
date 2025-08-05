import ollama
import os
import time

# Get all image paths from src/llvis directory
llvis_dir = "src/llvis"
image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
image_paths = []

for filename in os.listdir(llvis_dir):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        image_paths.append(f"{llvis_dir}/{filename}")

# Sort the paths for consistent ordering
image_paths.sort()
tem  = [item for item in image_paths if 'query' not in item]

print(f"Found {len(image_paths)} images:")
for path in image_paths:
    print(f"  {path}")

prompt_template = f" Analyze the reference image and the candidate image. Reference Image: src/llvis/query.jpg. Candidate Images: all images in {', '.join(tem)}"

prompt_template += """
    Task:
    1. Based on the subject, style, composition, and color, rate the similarity of the candidate image to the reference image on a scale of 0 to 100.
    2. Explain your reasoning for the similarity score.
    3. Selected Image path

    Format your response as a JSON object with the following keys: "similarity_score", "reasoning" and "candidate".
    """

start = time.time()

res = ollama.chat(
    model="qwen2.5vl:latest",
    messages=[{"role": "user", "content": prompt_template, "images": image_paths}],
)

end = time.time()

print(res["message"]["content"])

print(f"Time taken: {end - start:.2f} seconds for processing {len(image_paths)} images.")
