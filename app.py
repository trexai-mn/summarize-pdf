import openai
from dotenv import load_dotenv
import os
import PyPDF2

load_dotenv()  # Load environment variables from .env file

openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def summarize_text(text, max_tokens=300):
    prompt = f"Summarize the following text:\n\n{text}"
    response = openai.chat.completions.create(
        model="gpt-4o",  # or use "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()

# Main function
def summarize_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters from PDF.")

    if len(text) > 10000:
        text = text[:10000]  # truncate if too long
    summary = summarize_text(text)
    return summary

# Main usage
if __name__ == "__main__":
    file_name = "example.pdf"  # Replace with your PDF file path
    summary = summarize_pdf("example.pdf")  # Replace with your PDF file path

    print("\nPDF Summary:\n", summary)
