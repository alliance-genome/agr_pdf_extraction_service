import anthropic
from app.services.pdf_extractor import PDFExtractor

class LLM(PDFExtractor):
  def __init__(self, api_key, model='claude-sonnet-4-20250514'):
    self.client = client = anthropic.Anthropic(api_key=api_key)
    self.model = model

  def extract(self, grobid_md, docling_md, marker_md):
    prompt = self.create_prompt(grobid_md, docling_md, marker_md)

    try:
      message = self.client.messages.create(
          model=self.model,
          max_tokens=16000,
          messages=[
              {"role": "user", "content": prompt}
          ]
      )
      
      return message.content[0].text
    except Exception as e:
      raise Exception(f"Error in LLM processing: {str(e)}")

    return merged_md

  def create_prompt(self, grobid_md, docling_md, marker_md):
    return f"""You are processing a scientific article that has been extracted using three different tools: GROBID, Docling, and Marker. 
Each tool produces different quality outputs with varying levels of detail.

Your task is to:
1. Merge the three markdown extractions into a single, well-structured document
2. Identify and clearly mark the following sections:
   - Title
   - Authors (with affiliations if available)
   - Abstract
   - Keywords (if present)
   - Introduction
   - Methodology/Methods
   - Results
   - Discussion
   - Conclusion
   - References
   - Any other relevant sections

3. Extract and list:
   - All tables (preserve structure)
   - All figures/images (note their captions and references)
   - All equations (preserve formatting)

4. Resolve conflicts between the three extractions by choosing the most complete and accurate version
5. Maintain academic formatting and citation styles

Here are the three extractions:

## GROBID Extraction:
{grobid_md}

## Docling Extraction:
{docling_md}

## Marker Extraction:
{marker_md}

Please provide a single, well-structured markdown document with clear section headers and all elements properly organized."""


