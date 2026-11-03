"""Convert notebooks/*.py (percent format: '# %%' code cells, '# %% [markdown]' text cells) to .ipynb.
The .py files are the source of truth (clean diffs in git); the .ipynb files are generated, run them in Jupyter.
Usage: python tools/py2nb.py [notebooks/01_registration_plane_spacing.py ...]   (default: all)"""
import re, sys
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def convert(py):
    text = Path(py).read_text()
    chunks = re.split(r"(?m)^# %%(.*)$", text)
    cells = []
    for head, body in zip(chunks[1::2], chunks[2::2]):
        body = body.strip("\n")
        if "[markdown]" in head:
            cells.append(nbformat.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())))
        elif body.strip():
            cells.append(nbformat.v4.new_code_cell(body))
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    out = Path(py).with_suffix(".ipynb")
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    files = sys.argv[1:] or sorted((ROOT / "notebooks").glob("[0-9]*.py"))
    for f in files:
        print("wrote", convert(f))
