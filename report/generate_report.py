"""
Final Report and Code Package Builder (adapted from baselines/scripts/generate_doc.py)
CS 5134/6034 NLP Fall 2026 - Team FrameVision

Usage (from the repo root), after `python report/build_tables.py`:
    python report/generate_report.py [--final]
Builds report/FrameVision_Final_Report.{docx,pdf} (11 pt, 1-inch margins, at most 10 pages, PDF through
Microsoft Word) and report/FrameVision_Final_Code.zip (source, predictions and result JSONs, at most 10 MB).
Without --final, placeholders and missing figures are warnings; with --final they stop the build.
"""

import glob
import os
import re
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(REPO, "report")
MD_PATH = os.path.join(REPORT_DIR, "final_report.built.md")
DOCX_PATH = os.path.join(REPORT_DIR, "FrameVision_Final_Report.docx")
PDF_PATH = os.path.join(REPORT_DIR, "FrameVision_Final_Report.pdf")
ZIP_PATH = os.path.join(REPORT_DIR, "FrameVision_Final_Code.zip")
FONT = "Times New Roman"
BODY_PT = 11
MAX_PAGES = 10
TABLE_WIDTH_IN = 6.5
MAX_ZIP_BYTES = 10 * 1024 * 1024
ZIP_DIRS = ("src", "scripts", "tests", "evaluation", "ensemble", "analysis", "report", "baselines/src",
            "baselines/scripts", "baselines/tests", "docs", "predictions")
ZIP_EXTS = (".py", ".md", ".tsv", ".json", ".txt")
ZIP_ROOT_GLOBS = ("README.md", "requirements*.txt", "results/*.json", "baselines/results/*.json")
PLACEHOLDER = re.compile(r"\[\[(?:TBD|LEO)[^\]]*\]\]")
IMAGE = re.compile(r"!\[(.*?)\]\((.*?)\)")


def strip_comments(text):
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def final_problems(md_text, md_dir):
    """What must be fixed before the final build: placeholders left in the text and figures that do not exist."""
    problems = []
    placeholders = PLACEHOLDER.findall(md_text)
    if placeholders:
        shown = ", ".join(placeholders[:5]) + (", ..." if len(placeholders) > 5 else "")
        problems.append(f"{len(placeholders)} placeholder(s) left: {shown}")
    for _, path in IMAGE.findall(md_text):
        if not os.path.exists(os.path.join(md_dir, path)):
            problems.append(f"missing figure: {path}")
    return problems


def zip_members(root):
    """Repo-relative paths (forward slashes) that go into the code zip."""
    members = set()
    for pattern in ZIP_ROOT_GLOBS:
        for path in glob.glob(os.path.join(root, pattern)):
            if os.path.isfile(path):
                members.add(os.path.relpath(path, root).replace(os.sep, "/"))
    for folder in ZIP_DIRS:
        for current, dirs, files in os.walk(os.path.join(root, folder)):
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            for name in files:
                if name.endswith(ZIP_EXTS):
                    members.add(os.path.relpath(os.path.join(current, name), root).replace(os.sep, "/"))
    return sorted(members)


def add_inline(paragraph, text, size=BODY_PT):
    """Adds text with **bold**, *italic*, and `code` markup as formatted runs."""
    from docx.shared import Pt
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
    from docx.shared import Pt
    fmt = paragraph.paragraph_format
    fmt.space_after = Pt(after)
    fmt.space_before = Pt(before)
    fmt.line_spacing = 1.0


def set_cell_padding(table, left_right=50, top_bottom=0):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    margins = OxmlElement("w:tblCellMar")
    for side, width in (("top", top_bottom), ("left", left_right), ("bottom", top_bottom), ("right", left_right)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(width))
        node.set(qn("w:type"), "dxa")
        margins.append(node)
    table._tbl.tblPr.append(margins)


def add_table(doc, rows):
    from docx.shared import Inches, Pt
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


def build_docx(md_text, md_dir):
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    lines = strip_comments(md_text).splitlines()
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
            path = os.path.join(md_dir, IMAGE.match(line).group(2))
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            format_paragraph(paragraph, after=2)
            if os.path.exists(path):
                paragraph.add_run().add_picture(path, width=Inches(5.6))
            else:
                add_inline(paragraph, f"[missing figure: {os.path.relpath(path, md_dir)}]")
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
    from pypdf import PdfReader
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
    members = zip_members(REPO)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in members:
            archive.write(os.path.join(REPO, name), name)
    size = os.path.getsize(ZIP_PATH)
    print(f"Zip: {len(members)} files, {size / 1024:.1f} KB")
    if size > MAX_ZIP_BYTES:
        sys.exit("Zip is over the 10 MB limit.")


def main(argv):
    with open(MD_PATH, encoding="utf-8") as f:
        md_text = f.read()
    problems = final_problems(strip_comments(md_text), REPORT_DIR)
    for problem in problems:
        print(f"[generate_report] {problem}", file=sys.stderr)
    if problems and "--final" in argv:
        sys.exit("Final build stopped: fix the problems above.")
    build_docx(md_text, REPORT_DIR)
    build_pdf()
    build_zip()


if __name__ == "__main__":
    main(sys.argv[1:])
