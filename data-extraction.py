import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

def sanitize_filename(name):
    # Replace characters that are invalid on Windows
    return re.sub(r'[<>:"/\\|?*]', '_', name)

# Base URL of the meetings page
BASE_URLS = [
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjGcE7vQyaPX4JQEhAS-I4V_",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjGMJSsNGjDbhVRCqW1Bd221",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjFw3a9Z5rV4f2ZASkK1bSK2",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjE_yDbCZ40-XRLWm67GgcFd",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjE0Da_0Iki94symwVgHVBZm",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjG3NK-8H-KcSItYvXtVCfTM",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjHeD36UNQSGWetT70h6l2GH",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjHDlea_zImJLwfwgsnfaLZc",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjFTBcKPG5KUcT_vPZvVr_cP",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjHS8gs3CFGfTzsK0LKNmX9n",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjEJ1_Ljo-u8tLuNjKQJY0V_",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjFKQwcmQxXA0a57mbPI0jsa",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjEpjWYN9Hvo960DSD8Qdy0t",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjFDTebVthzJVdLhCPittEfM",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjGnyd5cwJXANpmjAcUPacSs",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjHav_jTzG4LbfG0zshScjL7",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjGu2c0qZIVI4gZ2rEu_UpC1",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjGqNlQ60gTwfuYIOmKpdI8T",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjHxaNb2UalBrvZN2DHeuc07",
    "https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjFIoAYGNENlfvgc1fDP5WuO"
]
# Directory to save downloaded PDFs
DOWNLOAD_DIR = "C:/Users/sophi/OneDrive - IU International University of Applied Sciences/IU/Project Data Analysis/Protokolle Stadtverordnetenversammlung"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Headers to mimic a browser visit
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


def get_meeting_links(base_url):
    meeting_links = []
    response = requests.get(base_url, headers=HEADERS)
    soup = BeautifulSoup(response.content, "html.parser")
    for link in soup.find_all("a", href=True):
        href = link['href']
        if "tops" in href:
            full_url = urljoin(base_url, href)
            meeting_links.append(full_url)
    return meeting_links


def download_protocols(meeting_links):
    for meeting_url in meeting_links:
        response = requests.get(meeting_url, headers=HEADERS)
        soup = BeautifulSoup(response.content, "html.parser")
        for link in soup.find_all("a", href=True):
            text = link.get_text(strip=True).lower()
            if "niederschrift" in text:
                pdf_url = urljoin(meeting_url, link['href'])
                raw_pdf_name = pdf_url.split("/")[-1]
                pdf_name = sanitize_filename(raw_pdf_name)
                pdf_path = os.path.join(DOWNLOAD_DIR, pdf_name)
                # Avoid redownloading
                if not os.path.exists(pdf_path):
                    pdf_response = requests.get(pdf_url, headers=HEADERS)
                    if pdf_response.status_code == 200:
                        with open(pdf_path, 'wb') as f:
                            f.write(pdf_response.content)
                        print(f"Downloaded: {pdf_name}")
                    else:
                        print(f"Failed to download: {pdf_name}")

def main():
    all_meeting_links = []
    for url in BASE_URLS:
        links = get_meeting_links(url)
        all_meeting_links.extend(links)
    print(f"Found {len(all_meeting_links)} meetings.")
    download_protocols(all_meeting_links)


if __name__ == "__main__":
    main()
