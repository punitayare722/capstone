import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

url = 'https://vcet.edu.in/'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

try:
    r = requests.get(url=url, headers=headers)
    r.raise_for_status()  # Raise an exception for bad status codes

    soup = BeautifulSoup(r.content, 'html.parser')

    # Extract href (links) data
    links = soup.find_all('a', href=True)
    print("Links:")
    for link in links:
        href = link['href']
        # Construct absolute URL if href is relative
        absolute_url = urljoin(url, href)
        print(absolute_url)

    # Extract PDF data
    print("\nPDF Links:")
    pdf_links = soup.find_all('a', href=lambda href: (href and href.endswith('.pdf')))
    for pdf_link in pdf_links:
        pdf_href = pdf_link['href']
        absolute_pdf_url = urljoin(url, pdf_href)
        print(absolute_pdf_url)

    # Extract text content from specific elements
    q = soup.find('a')
    if q:
        print("\nText Content:")
        print(q.get_text().strip())
    else:
        print("\nElement with class 'elementor-widget-container' not found")

except requests.exceptions.RequestException as e:
    print(f"Error fetching {url}: {e}")
from pdfminer.high_level import extract_text
import os

# Function to extract text from a PDF file using pdfminer.six
def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text

# Main function to process a PDF file
def process_pdf(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    print(pdf_text)  # Print the extracted text directly

# Example usage
if __name__ == "__main__":
    pdf_file = 'srsTemplate.pdf'  # Replace with your PDF file path
    process_pdf(pdf_file)
