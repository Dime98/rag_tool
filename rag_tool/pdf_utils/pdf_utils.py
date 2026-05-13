from pathlib import Path

import pymupdf


def extract_pdf_pages(pdf_folders: list[Path]) -> list[dict]:
    pages = []

    pdf_counter = 0
    for pdf in pdf_folders:
        pdf_counter += 1
        pdf_reader = pymupdf.open(pdf)

        for page_index, page in enumerate(pdf_reader, start=1):
            text = page.get_text().strip()
            if len(text) == 0:
                continue
            pages.append({"source": pdf.stem, "text": text, "page": page_index})

    return pages
