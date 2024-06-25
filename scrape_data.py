
#1 
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

url = 'https://vcet.edu.in/'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Function to save PDF files
def save_pdf(pdf_url, save_path='pdfs'):
    try:
        response = requests.get(pdf_url, headers=headers)
        response.raise_for_status()
        os.makedirs(save_path, exist_ok=True)
        pdf_filename = os.path.join(save_path, pdf_url.split('/')[-1])
        with open(pdf_filename, 'wb') as pdf_file:
            pdf_file.write(response.content)
        print(f"Saved PDF: {pdf_filename}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF {pdf_url}: {e}")

try:
    r = requests.get(url=url, headers=headers)
    r.raise_for_status()  # Raise an exception for bad status codes

    soup = BeautifulSoup(r.content, 'html.parser')

    # Extract href (links) data
    links = soup.find_all('a', href=True)
    print("Links and their content:")

    for link in links:
        href = link['href']
        # Construct absolute URL if href is relative
        absolute_url = urljoin(url, href)
        
        try:
            # Fetch the linked page
            linked_page_response = requests.get(absolute_url, headers=headers)
            linked_page_response.raise_for_status()

            # Parse the linked page content
            linked_soup = BeautifulSoup(linked_page_response.content, 'html.parser')
            
            # Extract and print the text content of the linked page
            linked_text = linked_soup.get_text(separator='\n', strip=True)
            print(f"\nURL: {absolute_url}")
            print(f"Content:\n{linked_text}\n")

        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve the linked page {absolute_url}: {e}")

    # Extract PDF data
    print("\nPDF Links:")
    pdf_links = soup.find_all('a', href=lambda href: (href and href.endswith('.pdf')))
    for pdf_link in pdf_links:
        pdf_href = pdf_link['href']
        absolute_pdf_url = urljoin(url, pdf_href)
        print(absolute_pdf_url)
        save_pdf(absolute_pdf_url)

except requests.exceptions.RequestException as e:
    print(f"Failed to retrieve the webpage: {e}")

    # Extract PDF datapp
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
from transformers import GPT2Tokenizer, GPT2Model
import torch

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Function to generate embeddings using GPT-2
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)  # Mean pooling of hidden states
    return embeddings

# Main function to process a PDF file
def process_pdf(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    embeddings = generate_embeddings(pdf_text)
    return embeddings
if __name__ == "__main__":
    pdf_files = [
        'FIRST-YEAR-ENGINEERING-ADMISSION-2023-2024-CAP-DOCUMENTS-FEE-STRUCTURE.pdf',
        'FIRST-YEAR-FEE-STRUCTURE-2023-24.pdf',
        'srsTemplate.pdf',
        # Add more PDF file paths as needed
    ]
    
    for pdf_file in pdf_files:
        embeddings = process_pdf(pdf_file)
        # Use embeddings as needed, e.g., save to file, perform further analysis
        print(f"Embeddings for {pdf_file}:")
        print(embeddings)

