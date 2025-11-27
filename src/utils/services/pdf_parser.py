import fitz
import re


class PDFParser:
    """
    A robust PDF parser that:
      - Extracts raw text from PDF pages
      - Skips Table of Contents (TOC) pages in the first few pages
      - Cleans text by removing headers, page numbers, and artifacts
      - Merges sentences split across page boundaries
    """

    def __init__(self, file_path, toc_check_pages=10):
        self.file_path = file_path
        self.toc_check_pages = toc_check_pages
        self.doc = None
        self.skipped_pages = []
        self.cleaned_pages = []

    def open_pdf(self):
        """Open the PDF document."""
        self.doc = fitz.open(self.file_path)

    def extract_text_from_pdf(self):
        """Extract raw text from all pages of the PDF."""
        if self.doc is None:
            self.open_pdf()

        pages = []
        for page_num, page in enumerate(self.doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page_number": page_num, "text": text})
        return pages

    @staticmethod
    def is_toc_page(text):
        """Detect if a page looks like a Table of Contents page."""
        lower_text = text.lower()
        if "contents" in lower_text or "table of contents" in lower_text:
            return True

        dotted_lines = sum(
            1 for line in text.splitlines() if re.search(r"\.{3,}\s*\d{1,3}$", line.strip())
        )
        return dotted_lines > 3

    @staticmethod
    def clean_page_text(text):
        """Remove standalone page numbers, headings, and unwanted artifacts."""
        cleaned_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # Remove isolated numbers, 'Page X', or 'Contents'
            if re.match(r"^\d{1,3}$", stripped):
                continue
            if re.match(r"^(page|Page)\s*\d{1,3}$", stripped):
                continue
            if stripped.lower() in ["contents", "table of contents"]:
                continue

            cleaned_lines.append(stripped)

        return " ".join(cleaned_lines).strip()

    @staticmethod
    def merge_cross_page_sentences(pages):
        """
        Merge sentences that continue across page boundaries.
        Handles cases like commas, colons, semicolons, dashes,
        or lack of ending punctuation.
        """
        merged = []
        skip_next = False

        # Pattern for pages that end mid-sentence or mid-clause
        CONTINUATION_PATTERN = re.compile(r"[,:;—–\-]$|[^.!?]['\"”’)]?\s*$")

        for i in range(len(pages) - 1):
            if skip_next:
                skip_next = False
                continue

            current = pages[i]["text"].strip()
            next_text = pages[i + 1]["text"].strip()

            # Merge if page ends mid-sentence or mid-clause
            if CONTINUATION_PATTERN.search(current):
                merged.append({
                    "page_start": pages[i]["page_number"],
                    "page_end": pages[i + 1]["page_number"],
                    "text": re.sub(r"\s+", " ", current + " " + next_text).strip()
                })
                skip_next = True
            else:
                merged.append({
                    "page_start": pages[i]["page_number"],
                    "page_end": pages[i]["page_number"],
                    "text": current
                })

        # Handle last page if it wasn’t merged
        if not skip_next and len(pages) > 0:
            merged.append({
                "page_start": pages[-1]["page_number"],
                "page_end": pages[-1]["page_number"],
                "text": pages[-1]["text"].strip()
            })

        return merged


    def parse_pdf(self):
        """
        Main pipeline:
        - Extracts raw text
        - Skips TOC pages (in the first few pages)
        - Cleans text
        - Merges split sentences across pages
        """
        raw_pages = self.extract_text_from_pdf()

        for p in raw_pages:
            if p["page_number"] <= self.toc_check_pages and self.is_toc_page(p["text"]):
                print(f"Skipping TOC page: {p['page_number']}")
                self.skipped_pages.append(p["page_number"])
                continue

            cleaned_text = self.clean_page_text(p["text"])
            self.cleaned_pages.append({"page_number": p["page_number"], "text": cleaned_text})

        merged_pages = self.merge_cross_page_sentences(self.cleaned_pages)

        print(f"Total pages after cleaning: {len(merged_pages)}")
        if self.skipped_pages:
            print(f"Skipped TOC pages (only in first {self.toc_check_pages}): {self.skipped_pages}")
        else:
            print("No TOC pages detected in the first few pages.")

        return merged_pages


