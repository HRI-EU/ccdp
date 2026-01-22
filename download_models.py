import os
import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/16VVNrU1PpgfjNibktpdD_tzZzdsSbSYn"
OUTPUT_DIR = "./models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

gdown.download_folder(
    url=FOLDER_URL,
    output=OUTPUT_DIR,
    quiet=False,
    use_cookies=False
)