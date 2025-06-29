import json
import os
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
from collections import defaultdict
import re

from google import genai
from google.genai import types
import PIL.Image
import copy
from google.genai.errors import ServerError
import time

def get_full_response_once(client, image, prompt, max_tokens=50000):
    """
    Single-shot call to Gemini Flash 2.0, requesting up to max_tokens.

    Arguments:
    - client:      Your google.genai client
    - image:       PIL Image (or binary) to include
    - prompt:      The text prompt for OCR
    - max_tokens:  Maximum tokens to request in the response

    Returns:
    - str:        The raw text response (with any ``` fences stripped)
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=[prompt, image],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_budget=1024
            )
        )
    )

    # Try the fields in order of preference:
    raw = None

    # 1) response.text (if the library actually populates it)
    raw = getattr(response, 'text', None)

    # 2) response.candidates[0].content (common in Google GenAI clients)
    if not raw and hasattr(response, 'candidates') and response.candidates:
        # pick the first candidate
        candidate = response.candidates[0]
        raw = getattr(candidate, 'content', None) or getattr(candidate, 'text', None)

    if not raw:
        raise RuntimeError(f"No content returned by Gemini (got: {response!r})")

    # Strip any ```json fences and return
    return raw.replace("```json\n", "").replace("```", "").strip()


def gemini_convert_to_json(images, page_num, client, folder_output, prompt, retry_delay=5):
    """
    Run Gemini Flash 2.5 for OCR and convert output to JSON, retrying on 503 errors.

    Arguments:
    - images (list): List of PIL Image objects
    - page_num (int): Page number of the report
    - client (object): Your Gemini API client
    - folder_output (str): Path to folder to store JSON/TXT files
    - prompt (str): The initial OCR prompt
    - retry_delay (int): Seconds to wait between retries on overload
    """
    # 1. Keep retrying until we get a response without a ServerError
    while True:
        try:
            full_output = get_full_response_once(
                client=client,
                image=images[page_num],
                prompt=prompt,
                max_tokens=50000
            )
            break
        except ServerError as e:
            # Only retry on 503 UNAVAILABLE
            msg = getattr(e, "status_code", None)
            if msg == 503 or "overloaded" in str(e).lower():
                print(f"[RETRY] Page {page_num+1}: model overloaded, retrying in {retry_delay}s…")
                time.sleep(retry_delay)
                continue
            # if it's some other server error, re-raise
            raise

    # 2. Define output paths
    json_output_path = os.path.join(folder_output, f"page_{page_num + 1}.json")
    txt_output_path  = os.path.join(folder_output, f"page_{page_num + 1}.txt")

    # 3. Attempt JSON parse
    try:
        json_data = json.loads(full_output)
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)
        print(f"[SUCCESS] Page {page_num+1} of {len(images)}: Created {json_output_path}\n")
    except json.JSONDecodeError:
        # 4. If JSON parsing fails, write raw text to a .txt file
        with open(txt_output_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(full_output)
        print(f"[FAIL]    Page {page_num+1} of {len(images)}: Invalid JSON. Saved as {txt_output_path}\n")

def run_OCR(report_path, prompt, folder_output, model='gemini-2.5-flash', api_key=None, display=False):
    """
    Run OCR for report

    Arguments:
    report_path (str): Path to report
    prompt (str): Prompt for OCR
    folder_output (str): Path to folder to store JSON files
    model (str): model_id (str): Model to use. Default 'gemini-flash-2.0'
    display (bool): Display images of page. Default False
    api_key (str): API key. If model is not open source, must provide API key. Default None

    Models that need API keys: gemini-flash-2.0, openai-gpt-4o, claude-sonnet-3.5

    Output:
    Created Excel and JSON file containing all samples

    json_list_by_sample (dict): List of JSON by sample name
    updated_json_list (dict): List of JSON by page
    """
    print("Processing report: ", report_path)

    # Create folder to store JSON
    os.makedirs(folder_output, exist_ok=True)

    # Convert PDF to image
    print("\nConverting PDF to image.")
    images = convert_from_path(report_path)

    # Display image
    if display:
        fig, axes = plt.subplots(1, 5, figsize=(15, 10))

        for i, ax in enumerate(axes.flat):
            img = np.array(images[i])
            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    # Count how many pages
    pages_count = len(images)

    # Check which OCR model is used
    if model in ('gemini-2.0-flash', 'gemini-2.5-flash'):
        client = genai.Client(api_key=api_key)

        # Loop OCR for every page
        print("\nRun OCR report")
        for page_num in range(pages_count):
            gemini_convert_to_json(images, page_num, client, folder_output, prompt)


    else:
        print(f'Model {model} currently not implemented')

def read_prompt(prompt_file_path):
  with open(prompt_file_path, 'r') as file:
    return file.read()
