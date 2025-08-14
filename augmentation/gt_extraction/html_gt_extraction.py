import os
import json
import re
import string
from bs4 import BeautifulSoup

def convert_arabic_to_gujarati(text):
    """
    Convert Arabic numerals in the input text to Gujarati numerals.
    """
    mapping = {
        "0": "૦",
        "1": "૧",
        "2": "૨",
        "3": "૩",
        "4": "૪",
        "5": "૫",
        "6": "૬",
        "7": "૭",
        "8": "૮",
        "9": "૯"
    }
    for arabic_digit, gujarati_digit in mapping.items():
        text = text.replace(arabic_digit, gujarati_digit)
    return text

def is_valid_entry(entry):
    """
    Returns True if the entry should be kept:
      - It is not of category "Sep".
      - Its text (after removing punctuation and extra spaces) is not empty.
    """
    # Remove entries with category "Sep" (case-insensitive)
    category = entry.get("category", "").strip().lower()
    if category == "sep":
        return False

    text = entry.get("text", "")
    # Remove punctuation using string.punctuation (for ASCII punctuation)
    cleaned_text = re.sub(r'[' + re.escape(string.punctuation) + r']', '', text)
    # Remove extra spaces
    cleaned_text = cleaned_text.strip()
    return bool(cleaned_text)

def process_json_file(input_path, output_path):
    """
    Load a JSON file (expected to be a list of dictionaries), filter out entries
    that are not valid (e.g. category "Sep" or text only with punctuation/extra spaces),
    and write the cleaned list to the output path.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter out entries that don't meet the criteria.
    filtered_data = [entry for entry in data if is_valid_entry(entry)]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
def process_folder(input_folder):
    """
    Process all JSON files in the given folder and save the cleaned versions to the same folder.
    """
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
            input_path = os.path.join(input_folder, file_name)
            try:
                process_json_file(input_path, input_path)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

def extract_ground_truth_from_html(html_file_path, output_dir):
    """
    Extracts ground truth text from a local HTML file and saves the text in JSON format for each image.
    
    The function reads an HTML file containing <div> elements with IDs starting with '_idContainer'.
    Each group starts when an <img> tag is encountered. The image's base name (without extension)
    is used to create a JSON file in the output directory. All subsequent text content (from
    <p>, <h1>, <h2>, <h3>, etc.) is grouped by their class and saved as a list of dictionaries.
    
    After extraction, Arabic numerals are replaced with Gujarati numerals in all the text entries.
    
    :param html_file_path: Path to the local HTML file.
    :param output_dir: Path to the output directory where results will be saved.
    """
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the HTML file
    with open(html_file_path, "r", encoding="utf-8") as f:
        html = f.read()
    
    # Parse the HTML
    soup = BeautifulSoup(html, "html.parser")
    
    # Find all div elements with id starting with "_idContainer" and sort them numerically.
    divs = soup.find_all("div", id=lambda x: x and x.startswith("_idContainer"))
    divs = sorted(divs, key=lambda d: int(d.get("id").replace("_idContainer", "")))
    
    # Group the content:
    # Each new <img> tag signals the start of a new group;
    # subsequent text elements are associated with that image.
    groups = []
    current_group = None
    
    for div in divs:
        img_tag = div.find("img")
        if img_tag:
            if current_group is not None:
                groups.append(current_group)
            current_group = {
                "img": img_tag.get("src"),
                "texts": []  # To store extracted text as dicts: {"category": ..., "text": ...}
            }
        else:
            if current_group is not None:
                # Extract text from target elements.
                text_tags = div.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
                for tag in text_tags:
                    cls = tag.get("class")
                    cls_name = cls[0] if cls else "unknown"
                    # Extract inner HTML, replace <br> with newline, then strip remaining tags.
                    html_content = tag.decode_contents()
                    html_content = re.sub(r'<br\s*/?>', '\n', html_content, flags=re.IGNORECASE)
                    text = BeautifulSoup(html_content, "html.parser").get_text(strip=True)
                    # Replace Arabic numerals with Gujarati numerals.
                    text = convert_arabic_to_gujarati(text)
                    current_group["texts"].append({"category": cls_name, "text": text})
    
    if current_group is not None:
        groups.append(current_group)
    
    # For each group, create a JSON file inside output_dir named after the image's base name,
    # and save the categorized text in JSON format.
    for group in groups:
        img_path = group["img"]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        json_file_path = os.path.join(output_dir, f"{base_name}.json")
        
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(group["texts"], f, ensure_ascii=False, indent=4)

    process_folder(output_dir)
    
    print("Extraction complete. Output saved in:", output_dir)


# Example usage:
if __name__ == "__main__":
    # Update these paths as needed.
    html_file_path = r"C:\Users\swapn\Downloads\Books\Swaminarayan Darshan\Sample01.html"
    output_dir = r"C:\Users\swapn\Downloads\OCR Experiments\Swaminarayan Darshan\ground_truth_extracted"
    extract_ground_truth_from_html(html_file_path, output_dir)