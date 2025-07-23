import openai
from dotenv import load_dotenv
import os
import PyPDF2
import click

load_dotenv()  # Load environment variables from .env file

openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def summarize_text(text, max_tokens=300, model="gpt-4.1"):
    prompt = f"Summarize giving text, use simple english, return only summary. text:\n\n{text}"

    try:
        response = openai.chat.completions.create(
            model=model,  # or use "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(e)
        return

    return response.choices[0].message.content.strip()


def summarize_pdf(pdf_file, model="gpt-4o"):
    text = extract_text_from_pdf(pdf_file)
    print(f"Summarizing PDF file - {pdf_file}")
    print(f"Extracted {len(text)} characters from PDF.")
    print("Sending text to OpenAI for summarization...")

    if not text:
        print("No text found in the PDF file.")
        return

    # TODO: Implement a check for text length and handle cases where the text is too long.
    summary = summarize_text(text, model=model)
    return summary


@click.command()
@click.argument("pdf_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", default='gpt-4o', help='OpenAI model to use (e.g., gpt-4o, gpt-3.5-turbo)')
def main(pdf_file, model):
    """
    Command line interface to summarize a PDF file.
    Usage: python app.py <path_to_pdf_file>
    """
    summary = summarize_pdf(pdf_file, model=model)
    if summary:
        print("Summary: \n" + summary)


# Main usage
if __name__ == "__main__":
    main()
