import re

# Drop a line only if the WHOLE line is header junk:
# titles, dates (3/14/16), codes (OM-500, 16-03), page counts (1 of 2)
SKIP = re.compile(
    r"^(OMH Official Policy Manual|Official Policy Manual|Date issued|Date Issued|"
    r"T\.L\.|Section #|Section:|Directive:|Policy Owner:|Introduction|Preface|"
    r"Operational Management|Patient Care|"
    r"Page( \d+ of \d+)?|_+|"                                   # <- new: "Page" and "____" lines
    r"\d{1,2}/\d{1,2}/\d{2,4}|\d+ of \d+|(I|OM|PC)-\d+|\d{2}-\d{2})$",
    re.IGNORECASE,
)


def clean_page(text):
    lines = [l.strip() for l in text.split("\n")]
    kept = [l for l in lines if l and not SKIP.match(l)]
    return "\n".join(kept)


if __name__ == "__main__":
    from load_pdfs import load_pdfs
    for p in load_pdfs()[:3]:
        print("=====", p["source"], "page", p["page"])
        print(clean_page(p["text"])[:500])