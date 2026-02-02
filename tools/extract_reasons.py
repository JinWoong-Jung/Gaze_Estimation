import os
import base64
import time
import cv2
import pandas as pd
import argparse
from pathlib import Path
from openai import OpenAI
import numpy as np

def encode_image_from_file(image_path: str) -> str:
    """Encodes an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def encode_image_from_memory(img_array: np.ndarray, ext: str = ".jpg") -> str:
    """Encodes an OpenCV image (numpy array) to base64 string."""
    _, buffer = cv2.imencode(ext, img_array)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")

def guess_mime(filename: str) -> str:
    """Guesses MIME type from file extension."""
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".png":
        return "image/png"
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    return "application/octet-stream"

def load_annotations(annotation_path: str) -> pd.DataFrame:
    """Loads annotations from CSV file."""
    columns = ["path", "id", "body_x", "body_y", "body_w", "body_h",
               "eye_x", "eye_y", "gaze_x", "gaze_y",
               "head_xmin", "head_ymin", "head_xmax", "head_ymax",
               "inout", "origin", "meta"]

    try:
        df = pd.read_csv(annotation_path, sep=",", names=columns,
                         index_col=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return pd.DataFrame()
    return df

def create_text_coordinate_prompt(row: pd.Series, image_width: int, image_height: int) -> str:
    """Generates the prompt text based on annotation and image size."""
    head_xmin = int(float(row['head_xmin']))
    head_ymin = int(float(row['head_ymin']))
    head_xmax = int(float(row['head_xmax']))
    head_ymax = int(float(row['head_ymax']))

    gaze_x = int(row['gaze_x'] * image_width)
    gaze_y = int(row['gaze_y'] * image_height)

    coord_text = f"""Task:
You are tasked with determining the gaze target where person is looking at and its reasoning in a provided image using spatial information: image size, the person's head bounding box, and their gaze target coordinates.

Spatial Information:
- Image size (width x height) = ({image_width} x {image_height})
- Person's Head Bbox [xmin, ymin, xmax, ymax] = [{head_xmin}, {head_ymin}, {head_xmax}, {head_ymax}]
- Gaze target point (x, y) = ({gaze_x}, {gaze_y})

Methods:
1. Use the pixel at the gaze target as main evidence. Only consider context or surroundings if the pixel is not sufficient.
2. Produce a concise noun phrase as the object/area prediction.
3. Explain why the point is inferred using visible cues.

Output Format:
Object: <Concise noun phrase for gaze target>
Reasoning: <A brief reasoning in 1-2 sentences. You have to answer as if you were not given any coordinates.>"""
    return coord_text

def create_marked_image(img: np.ndarray, row: pd.Series) -> np.ndarray:
    """Creates a marked version of the image with annotations drawn."""
    marked_img = img.copy()
    h, w = marked_img.shape[:2]

    head_xmin = int(float(row['head_xmin']))
    head_ymin = int(float(row['head_ymin']))
    head_xmax = int(float(row['head_xmax']))
    head_ymax = int(float(row['head_ymax']))
    cv2.rectangle(marked_img, (head_xmin, head_ymin), (head_xmax, head_ymax),
                  (255, 255, 0), 2)

    eye_x = int(row['eye_x'] * w)
    eye_y = int(row['eye_y'] * h)

    gaze_x = int(row['gaze_x'] * w)
    gaze_y = int(row['gaze_y'] * h)

    cv2.circle(marked_img, (gaze_x, gaze_y), 5, (0, 0, 255), -1)
    cv2.arrowedLine(marked_img, (eye_x, eye_y), (gaze_x, gaze_y),
                    (0, 255, 255), 2)

    return marked_img

def process_pipeline(args):
    """Main processing pipeline."""
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("Warning: No API key provided. Set OPENAI_API_KEY env variable or use --api_key argument.")

    client = OpenAI(api_key=api_key)

    # Load annotations
    print(f"Loading annotations from {args.annotation}...")
    df = load_annotations(args.annotation)
    if df.empty:
        print("No annotations loaded. Exiting.")
        return

    print(f"=== Reasoning Effort: {args.reasoning_effort}, Verbosity: {args.verbosity} ===")
    print(f"Processing {len(df)} annotations...")

    # Statistics
    total_input_tokens = 0
    total_output_tokens = 0
    processed_count = 0
    skipped_count = 0
    total_processing_time = 0.0

    # Resume logic: Scan existing outputs
    print("Scanning existing outputs to resume...")
    processed_ids = set()
    output_base_dir = os.path.join(args.output_root, "output")
    
    if os.path.exists(output_base_dir):
        for root, dirs, files in os.walk(output_base_dir):
            for file in files:
                if file.endswith(".txt"):
                    # Filename format: {basename}_{id}.txt
                    try:
                        name_without_ext = os.path.splitext(file)[0]
                        annot_id = name_without_ext.split('_')[-1]
                        processed_ids.add(annot_id)
                    except:
                        pass
    
    print(f"Found {len(processed_ids)} already processed annotations. Resuming...")

    for idx, (index, row) in enumerate(df.iterrows(), 1):
        row_id = str(row['id'])
        
        # Skip if already processed
        if row_id in processed_ids:
            continue

        rel_path = row['path']
        image_path = os.path.join(args.image_root, rel_path)

        if not os.path.exists(image_path):
            print(f"[{idx}/{len(df)}] Skipped: Image not found {image_path}")
            skipped_count += 1
            continue

        # Define output paths preserving structure
        # Output root structure:
        # gazefollow_vlm/
        #   output/train/000000/img_id.txt
        #   mark/train/000000/img_id.jpg
        #   prompt/train/000000/img_id.txt
        
        rel_dir = os.path.dirname(rel_path)
        basename = os.path.splitext(os.path.basename(rel_path))[0]
        row_id = str(row['id'])
        
        out_dir_base = os.path.join(args.output_root, "output", rel_dir)
        mark_dir_base = os.path.join(args.output_root, "mark", rel_dir)
        prompt_dir_base = os.path.join(args.output_root, "prompt", rel_dir)
        
        # Output filenames with ID to handle multiple annotations per image
        out_file_name = f"{basename}_{row_id}.txt"
        mark_file_name = f"{basename}_{row_id}.jpg"
        prompt_file_name = f"{basename}_{row_id}.txt"
        
        out_path = os.path.join(out_dir_base, out_file_name)
        
        # Skip if already exists
        if os.path.exists(out_path):
            # print(f"[{idx}/{len(df)}] Skipped: Already processed {out_file_name}")
            continue

        # Create directories only when needed
        Path(out_dir_base).mkdir(parents=True, exist_ok=True)
        if args.save_mark: Path(mark_dir_base).mkdir(parents=True, exist_ok=True)
        if args.save_prompt: Path(prompt_dir_base).mkdir(parents=True, exist_ok=True)

        # Load Original Image
        img = cv2.imread(image_path)
        if img is None:
            print(f"[{idx}/{len(df)}] Skipped: Failed to load image {image_path}")
            skipped_count += 1
            continue
            
        h, w = img.shape[:2]

        # 1. Prepare Prompt
        prompt_text = create_text_coordinate_prompt(row, w, h)

        if args.save_prompt:
            with open(os.path.join(prompt_dir_base, prompt_file_name), "w", encoding="utf-8") as f:
                f.write(prompt_text)

        # 2. Prepare Marked Image
        marked_img = create_marked_image(img, row)

        if args.save_mark:
            cv2.imwrite(os.path.join(mark_dir_base, mark_file_name), marked_img)

        # 3. Prepare Base64 inputs
        base64_image = encode_image_from_file(image_path)
        mime_type = guess_mime(image_path)
        data_url = f"data:{mime_type};base64,{base64_image}"

        base64_mark = encode_image_from_memory(marked_img, os.path.splitext(image_path)[1])
        data_url_mark = f"data:{mime_type};base64,{base64_mark}"

        print(f"[{idx}/{len(df)}] Processing {rel_path} (ID: {row_id})...")

        # 4. Call GPT API
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            start_time = time.time()
            try:
                resp = client.responses.create(
                    model="gpt-5.2",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt_text},
                                {"type": "input_image", "image_url": data_url},
                                {"type": "input_image", "image_url": data_url_mark},
                            ],
                        }
                    ],
                    reasoning={"effort": args.reasoning_effort},
                    text={"verbosity": args.verbosity},
                    max_output_tokens=2000,
                )

                result_text = resp.output_text or ""
                processed_result_text = "\n".join([line for line in result_text.splitlines() if line.strip()])

                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(processed_result_text)

                usage = getattr(resp, "usage", None)
                in_tok = getattr(usage, "input_tokens", 0) if usage else 0
                out_tok = getattr(usage, "output_tokens", 0) if usage else 0
                total_input_tokens += in_tok
                total_output_tokens += out_tok

                end_time = time.time()
                processing_time = end_time - start_time
                total_processing_time += processing_time
                processed_count += 1

                print(f"  -> Success: in={in_tok}, out={out_tok}, time={processing_time:.2f}s")
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  -> Retry {attempt + 1}/{max_retries} due to error: {e}")
                    time.sleep(retry_delay)
                else:
                    print(f"  -> Error processing {rel_path} ID {row_id}: {e}")
                    skipped_count += 1

    # Final Summary
    avg_processing_time = total_processing_time / processed_count if processed_count > 0 else 0
    input_cost = (total_input_tokens / 1_000_000) * 1.75
    output_cost = (total_output_tokens / 1_000_000) * 14.0
    total_cost = input_cost + output_cost
    avg_total_cost = total_cost / processed_count if processed_count > 0 else 0

    print("\n" + "=" * 50)
    print(f"Total processed: {processed_count}")
    print(f"Total skipped: {skipped_count}")
    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Average processing time: {avg_processing_time:.2f}s")
    print(f"Estimated Cost: ${total_cost:.4f}")
    print(f"Results saved to: {args.output_root}")
    print("=" * 50)

    summary_path = os.path.join(args.output_root, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== GPT Processing Summary ===\n\n")
        f.write(f"1. Reasoning Effort: {args.reasoning_effort}\n")
        f.write(f"2. Verbosity Level: {args.verbosity}\n")
        f.write(f"3. Input tokens/Cost: {total_input_tokens:,} / ${input_cost:.6f}\n")
        f.write(f"4. Output tokens/Cost: {total_output_tokens:,} / ${output_cost:.6f}\n")
        f.write(f"5. Total Cost: ${total_cost:.6f}\n")
        f.write(f"6. Avg Cost per Image: ${avg_total_cost:.6f}\n")

    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with GPT-5.2 (VLM Prompting)")

    # Paths
    parser.add_argument("--annotation", type=str, default="data/gazefollow/train_annotations_new.txt",
                        help="Path to annotation file")
    parser.add_argument("--image_root", type=str, default="data/gazefollow_extended",
                        help="Root directory containing images (referenced by relative paths in annotation)")
    parser.add_argument("--output_root", type=str, default="data/gazefollow_reason",
                        help="Root directory for outputs (will contain output, mark, prompt subdirs)")

    # Flags
    parser.add_argument("--save_mark", action="store_true", default=True,
                        help="Whether to save the marked images")
    parser.add_argument("--save_prompt", action="store_true", default=True,
                        help="Whether to save the generated prompt files")

    # API & Model params
    parser.add_argument("--api_key", type=str, required=True,
                        help="OpenAI API key")
    parser.add_argument("--reasoning_effort", type=str, default="medium",
                        choices=["none", "low", "medium", "high"])
    parser.add_argument("--verbosity", type=str, default="medium",
                        choices=["low", "medium", "high"])

    args = parser.parse_args()

    # Ensure output root exists
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    process_pipeline(args)