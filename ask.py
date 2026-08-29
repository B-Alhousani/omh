import sys
from collections import defaultdict

import ollama

from config import CHAT_MODEL, TOP_K
from search import load_retriever

TITLES = {
    "i-100intranet.pdf": "Preface (I-100)",
    "om-500.pdf": "Acceptable Use of Internet Policy (OM-500)",
    "om-505.pdf": "E-Mail Policy (OM-505)",
    "pc-522.pdf": "Staff and Client Use of Shared iPads (PC-522)",
}

NOT_FOUND = "I could not find this in the provided policies."

PROMPT = """You are a helpful assistant answering questions about OMH policies.
Answer ONLY using the context below. If the answer is not in the context, say
"{not_found}"

Rules:
- Answer in 1-3 sentences. No preamble, no quotes from the context.
- Start directly with the answer.
- Do not cite sources or name documents. They are listed separately.

Context:
{context}

Question: {question}

Answer:"""


def format_sources(hits):
    """Group retrieved pages by document. Always true: comes from metadata."""
    pages = defaultdict(set)
    for h in hits:
        pages[h["source"]].add(h["page"])
    lines = []
    for src in sorted(pages):
        nums = sorted(pages[src])
        label = "page" if len(nums) == 1 else "pages"
        title = TITLES.get(src, src)
        lines.append(f"- {title} - {label} {', '.join(str(n) for n in nums)}")
    return "\n".join(lines)


def _is_refusal(answer):
    return NOT_FOUND.rstrip(".").lower() in answer.lower()


def ask(question, retriever):
    hits = retriever.search(question, k=TOP_K)
    context = "\n\n".join(
        f"[{TITLES.get(h['source'], h['source'])}]\n{h['text']}" for h in hits
    )
    prompt = PROMPT.format(
        context=context, question=question, not_found=NOT_FOUND
    )
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = resp["message"]["content"]
    if _is_refusal(answer):
        return answer, []
    return answer, hits


def print_answer(answer, hits):
    print(answer)
    if hits:
        print("\nSources used to answer:")
        print(format_sources(hits))


def repl(retriever):
    print('Ask about the OMH policies. Type "exit" to quit.')
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question in {"exit", "quit"}:
            return
        if question:
            print_answer(*ask(question, retriever))


if __name__ == "__main__":
    retriever = load_retriever()
    if len(sys.argv) > 1:
        print_answer(*ask(sys.argv[1], retriever))
    else:
        repl(retriever)