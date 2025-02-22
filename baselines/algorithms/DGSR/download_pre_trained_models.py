import io
import os
import zipfile

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Replace with your own Google Drive zip folder public URL
ZIP_FOLDER_URL = "https://drive.google.com/file/d/1tZcyDKT2GeP35fTnnddiiUC3ow6OVcIF/view?usp=sharing"


def get_download_url(zip_folder_url):
    file_id = zip_folder_url.split("/")[-2]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return download_url


def bypass_virus_scan_warning(response):
    soup = BeautifulSoup(response.text, "html.parser")
    confirm_form = soup.find("form", id="download-form")
    if confirm_form:
        action = confirm_form.get("action")  # pyright: ignore
        return action
    return None


def download_zip_folder(download_url):
    response = requests.get(download_url, stream=True)  # pylint: disable=missing-timeout
    if "virus scan warning" in response.text.lower():
        bypass_url = bypass_virus_scan_warning(response)
        if bypass_url:
            # pylint: disable-next=missing-timeout
            response = requests.get(bypass_url, stream=True)  # pyright: ignore
    return response


def extract_zip_folder(zip_data, output_folder):
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_file:
        zip_file.extractall(output_folder)
    print(f"Extracted files to: {output_folder}")


def download_pre_trained_models_and_results():
    download_url = get_download_url(ZIP_FOLDER_URL)
    response = download_zip_folder(download_url)

    if response.status_code == 200:
        # Get the total size of the zip folder from the response headers
        total_size = int(response.headers.get("content-length", 0))

        # Download the zip folder with progress bar
        chunk_size = 1024
        zip_data = io.BytesIO()
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading", ncols=100) as pbar:
            for chunk in response.iter_content(chunk_size):
                zip_data.write(chunk)
                pbar.update(len(chunk))

        output_folder = os.getcwd()
        extract_zip_folder(zip_data.getvalue(), output_folder)
    else:
        print("Failed to download the zip folder.")


if __name__ == "__main__":
    download_pre_trained_models_and_results()
