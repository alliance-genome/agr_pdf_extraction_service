class PDFExtractor:
    def extract(self, pdf_path, output_filename):
        """
        Extract content from PDF and export to the specified format.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")