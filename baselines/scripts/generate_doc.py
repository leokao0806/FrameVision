"""
Report and Submission Package Builder
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao, En-Shuo Zhang

Intermediate_Report.md is the single source of the report text. This script builds:
1. Intermediate_Report.docx (11pt font, 1-inch margins)
2. Intermediate_Report.pdf  (through Microsoft Word, with a page-limit check)
3. FrameVision_Intermediate_Code.zip (source code only, 10 MB limit)
"""

import os
import re
import subprocess
import sys
import zipfile

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from pypdf import PdfReader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(project_root, "Intermediate_Report.md")
DOCX_PATH = os.path.join(project_root, "Intermediate_Report.docx")
PDF_PATH = os.path.join(project_root, "Intermediate_Report.pdf")
ZIP_PATH = os.path.join(project_root, "FrameVision_Intermediate_Code.zip")
FONT = "Times New Roman"
BODY_PT = 11
MAX_PAGES = 4
TABLE_WIDTH_IN = 6.5
MAX_ZIP_BYTES = 10 * 1024 * 1024
ZIP_DIRS = ("src", "scripts", "tests")
ZIP_RESULT_FILES = ("subtask1_metrics.json", "subtask2_metrics.json", "vlm_dev_metrics.json",
                    "vlm_test_metrics.json", "leakage_analysis.json", "course_split.json", "comparisons.json")
ZIP_ROOT_FILES = ("README.md", "requirements.txt", "requirements.lock.txt")


def add_inline(paragraph, text, size=BODY_PT):
    """Adds text with **bold**, *italic*, and `code` markup as formatted runs."""
    for token in re.split(r"(\*\*.+?\*\*|\*.+?\*|`.+?`)", text):
        if not token:
            continue
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
        elif token.startswith("*"):
            run = paragraph.add_run(token[1:-1])
            run.italic = True
        else:
            run = paragraph.add_run(token)
        run.font.name = FONT
        run.font.size = Pt(size)


def format_paragraph(paragraph, after=3, before=0):
    fmt = paragraph.paragraph_format
    fmt.space_after = Pt(after)
    fmt.space_before = Pt(before)
    fmt.line_spacing = 1.0


def set_cell_padding(table, left_right=50, top_bottom=0):
    tbl_pr = table._tbl.tblPr
    margins = OxmlElement("w:tblCellMar")
    for side, width in (("top", top_bottom), ("left", left_right), ("bottom", top_bottom), ("right", left_right)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(width))
        node.set(qn("w:type"), "dxa")
        margins.append(node)
    tbl_pr.append(margins)


def add_table(doc, rows):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    set_cell_padding(table)
    # Columns are sized by their longest cell, and rows stay together on one page.
    table.autofit = False
    lengths = [max(len(row[c]) for row in rows) + 5 for c in range(len(rows[0]))]
    widths = [TABLE_WIDTH_IN * length / sum(lengths) for length in lengths]
    for r, cells in enumerate(rows):
        for c, text in enumerate(cells):
            cell = table.cell(r, c)
            cell.width = Inches(widths[c])
            paragraph = cell.paragraphs[0]
            format_paragraph(paragraph, after=0)
            paragraph.paragraph_format.keep_with_next = r < len(rows) - 1
            add_inline(paragraph, f"**{text}**" if r == 0 and text else text)
    spacer = doc.add_paragraph()
    format_paragraph(spacer, after=0)
    spacer.paragraph_format.line_spacing = Pt(5)


def build_docx():
    with open(MD_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()

    doc = docx.Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.left_margin = section.right_margin = section.top_margin = section.bottom_margin = Inches(1.0)
    doc.styles["Normal"].font.name = FONT
    doc.styles["Normal"].font.size = Pt(BODY_PT)

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line or line == "---":
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            add_table(doc, rows)
        elif line.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            format_paragraph(paragraph, after=2)
            paragraph.add_run().add_picture(os.path.join(project_root, match.group(2)), width=Inches(5.6))
            i += 1
        elif line.startswith("# "):
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            format_paragraph(paragraph, after=2)
            add_inline(paragraph, f"**{line[2:]}**", size=14)
            i += 1
        elif line.startswith("## "):
            paragraph = doc.add_paragraph()
            format_paragraph(paragraph, after=2, before=5)
            paragraph.paragraph_format.keep_with_next = True
            add_inline(paragraph, f"**{line[3:]}**", size=12)
            i += 1
        elif line.startswith("- "):
            paragraph = doc.add_paragraph()
            format_paragraph(paragraph, after=2)
            paragraph.paragraph_format.left_indent = Inches(0.18)
            paragraph.paragraph_format.first_line_indent = Inches(-0.18)
            add_inline(paragraph, "• " + line[2:])
            i += 1
        elif line.startswith("^"):
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            format_paragraph(paragraph, after=2)
            add_inline(paragraph, line[1:].strip())
            i += 1
        else:
            paragraph = doc.add_paragraph()
            format_paragraph(paragraph)
            add_inline(paragraph, line)
            i += 1
    doc.save(DOCX_PATH)


def build_pdf():
    command = (
        "$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
        f"$d = $w.Documents.Open('{DOCX_PATH}', $false, $true); "
        f"$d.SaveAs2('{PDF_PATH}', 17); $d.Close(0); $w.Quit()"
    )
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command], check=True, timeout=180)
    pages = len(PdfReader(PDF_PATH).pages)
    print(f"PDF pages: {pages}")
    if pages > MAX_PAGES:
        sys.exit(f"Report is {pages} pages; the limit is {MAX_PAGES}.")


def build_zip():
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ZIP_ROOT_FILES:
            archive.write(os.path.join(project_root, name), name)
        for name in ZIP_RESULT_FILES:
            archive.write(os.path.join(project_root, "results", name), f"results/{name}")
        for folder in ZIP_DIRS:
            for current, _, files in os.walk(os.path.join(project_root, folder)):
                for name in files:
                    if name.endswith(".py"):
                        path = os.path.join(current, name)
                        archive.write(path, os.path.relpath(path, project_root))
    size = os.path.getsize(ZIP_PATH)
    print(f"Zip size: {size / 1024:.1f} KB")
    if size > MAX_ZIP_BYTES:
        sys.exit("Zip is over the 10 MB limit.")


if __name__ == "__main__":
    build_docx()
    build_pdf()
    build_zip()
