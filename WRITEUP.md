# WRITEUP

## What I built

A RAG pipeline that answers questions about 4 OMH policy PDFs, with
citations. Everything runs local: no API keys, no cloud, no cost. Anyone
can clone the repo and run it with Ollama installed.

Pipeline: load PDFs -> clean text -> chunk -> embed -> hybrid search ->
rerank -> LLM answers with citations.

21 pages, 40 chunks.

## Tools and why

**Ollama (local LLM), instead of OpenAI/Claude API.**
The documents are health policies, so I kept everything on the machine: no
data leaves, no key, no cost. The reviewer can run the repo without any
account. Trade-off: llama3.2 (3B) is much weaker than the big API models.
I accepted that and kept the model's job small: read 6 chunks, answer,
cite the document name. For this task it is enough.

**nomic-embed-text for embeddings.**
It runs in Ollama, so one runtime for the LLM and the embeddings. Local and
free, same privacy argument. Other open models (bge, e5) need
sentence-transformers and torch already at index time; nomic through Ollama
was the simplest path. For this corpus, embedding quality was not the
bottleneck. Chunking and keyword matching were (see below).

**ChromaDB, instead of numpy, FAISS, or a cloud vector DB.**
- numpy: works for 40 chunks, but I would build metadata storage and disk
  persistence by hand. The metadata (file, page) is what makes citations
  work.
- FAISS: an index, not a store. No metadata, no persistence, harder on
  Windows. Its speed matters at millions of vectors; I have 40.
- Pinecone/Weaviate: cloud, keys, accounts. That kills the "clone and run"
  goal.
- Chroma: metadata, persistence and cosine distance in a few lines, running
  inside the Python process. Right size for the task.

**rank_bm25 for keyword search.**
I added it after semantic search failed on exact terms (the "Gmail" story
below). Elasticsearch also gives BM25, but it is a server to install,
overkill for 40 chunks. rank_bm25 is a small pure-Python library. It is
naive and slow at scale, which does not matter here.

**Cross-encoder reranker (ms-marco-MiniLM-L6-v2).**
Without it, the right chunk often ranked 7 to 15, outside my top 6. The
cross-encoder reads question and chunk together, so it judges relevance
better than embeddings, which encode them separately. Reranking with the
LLM would also work, but that is 20 slow LLM calls per question. This model
is 90MB and fast on CPU for 20 pairs.

## PDF cleaning

Every page repeats a header table: "Official Policy Manual", dates, section
codes (OM-500), "Page x of y". 21 pages, same junk. Left in, it pollutes
every chunk and hurts search.

The cleaner drops a line only if the whole line is header junk (one regex).
Real sentences never match the pattern, so nothing important is lost. I
kept the policy title and owner lines on purpose. They are good search
keywords ("Acceptable Use of Internet Policy").

It took two iterations: my first pattern missed "Page 4 of 5" on one line
and the "____" underscore lines. I found this by reading the actual chunks
that search returned. They started with "Page 4 of 5".

Known issue I did not fix: pypdf breaks some words ("essentia l",
"requirement s"). A safe fix needs a dictionary, which is more machinery
than this task needs. It did not block correct answers in my tests.

## Chunking

Sliding window: 1500 characters, 300 overlap.

I started with 800/150 and hit a real failure: the "banned websites" list
got separated from its heading "Internet access misuse includes". The model
saw "Hotmail, Gmail or Yahoo mail" without knowing this is a ban list, and
correctly answered "I could not find this". Bigger chunks keep a heading
and its items together. The overlap protects sentences cut at the border.

A better approach is splitting at section headings ("1)", "(a)", "i.").
I skipped it because the 4 PDFs use different layouts and the regex would
be fragile. It is my first "what's next" item.

## Retrieval: what broke and how I fixed it

This was the main debugging story of the project.

**v1: pure embedding search.** Failed on: "Can staff use Gmail at work?"
The answer lives in a list. A list of many unrelated banned things has a
weak "average meaning", so its vector sits far from the question. Embedding
search is weak on exact rare words, the opposite of keyword search.

**v2: hybrid.** BM25 and embeddings, merged with Reciprocal Rank Fusion
(each engine ranks all chunks; a chunk's score is 1/(10+rank_a) +
1/(10+rank_b)). I used ranks, not raw scores, because cosine distance and
BM25 scores are different units. Along the way I fixed a tokenizer bug:
"Gmail," with a comma did not match "gmail". BM25 needs punctuation
stripped.

**v3: reranker on top.** Hybrid takes 20 candidates, the cross-encoder
re-orders them, the top 6 go to the LLM. This is the standard
retrieve->rerank pattern, and it is what finally put the right chunk in the
context.

I also tried query expansion: the LLM rewrites the question into policy
keywords before searching. It helped in v2 and became unnecessary in v3, so
I removed it.

**Sources printed under a refusal.** Asked "What is the capital of France",
the model correctly said it could not find the answer, and the code still
printed a Sources block underneath. Retrieval always returns exactly k
chunks; there is no "nothing matched" state, so it returned the 6 least bad
chunks for a question about France. Fixing that in retrieval would mean a
relevance cutoff on the reranker score, and those scores do not separate
cleanly: a genuinely relevant question scored -9.9 and pure nonsense -10.9,
one point apart. No honest threshold lives in that gap. So the fix sits
where the signal is reliable: the refusal sentence is a constant injected
into the prompt, `ask()` returns no hits when the answer contains it, and
the printer skips the Sources block when there are no hits.

## Citations

The model does not write citations at all. The code prints "Sources used to
answer" under the reply, with document titles and page numbers taken from
chunk metadata. That part cannot be hallucinated; it comes from pypdf page
numbers carried through the whole pipeline.

I arrived at this by testing. My first version asked the model to name the
source document in brackets after each fact, on the theory that with only 4
documents it would be hard to get wrong. Running 16 questions across the
four policies showed otherwise: roughly a third of the brackets named the
wrong document. "Are e-mail messages considered state records?" was answered
correctly from OM-505 and cited OM-500. "Who does the policy manual apply
to?" was answered from I-100 and cited OM-500. The model also invented
section labels that appear nowhere in the context, such as [iPad Sanitation
Guidelines] and [OM-500, D. Body of Directive, 1) Acceptable Use].

The Sources block was right in all 16. So the model's citations were adding
a second, less reliable answer to a question the metadata already answered.
Removing them also made several answers better: asked whether a personal
e-mail account may be used for OMH business, the citing version replied with
a statement about Internet access being a privilege, and the version without
citations correctly said to use the OMH e-mail service.

## Code structure

One file per pipeline stage, chained by a single dict, `{source, page,
text}`, that survives from pypdf all the way to the printed citation. That
contract is what makes the page numbers trustworthy.

`config.py` owns every tunable value: chunk size, overlap, model names,
batch size, how many chunks reach the reranker and how many reach the
prompt. The embedding model name is the one that matters most. Indexing and
search have to use the same model, and if they disagree retrieval gets
quietly worse with no error, so both sides import it from one place.

`search.py` exposes a `Retriever` that receives its four collaborators as
constructor arguments: the Chroma collection, the chunks, the BM25 index and
the cross-encoder. `load_retriever()` is the only function that knows how to
build them. Importing the module costs nothing, needs no database on disk,
and a test can build a `Retriever` out of fakes.

`build_index.py` splits into `build_chunks`, `fresh_collection`,
`index_chunks` and `build`. `index_chunks` yields how many chunks are done
rather than printing progress itself, so `build` decides what progress looks
like.

## Tests

36 unit tests, about 5 seconds, no database and no models. The retriever's
collaborators are replaced by fakes: a collection that records what was
written to it, a reranker that returns fixed scores, a stubbed
`ollama.embed`.

They cover the places where a silent bug is most likely: that chunk overlap
actually overlaps, that RRF prefers a chunk both engines rank well over one
with a single strong vote, that batch ids stay unique and sequential across
a batch boundary, that pages group correctly under one document, and that a
refusal returns no sources.

What they do not cover: whether Ollama and Chroma accept what I send them.
That only comes from running `build_index.py` for real.

## Known limits

- llama3.2 occasionally names a document inside the answer even though the
  prompt tells it not to. It is rare and no longer wrong when it happens,
  but the instruction is not obeyed every time.
- Answers vary between runs at the same settings. The Gmail question was
  answered correctly in 4 of 5 runs; the fifth drifted onto an unrelated
  passage about AI risk. Nothing in the pipeline pins the sampling.
- Broken words from pypdf remain in the text.
- Retrieval brings some irrelevant chunks next to the right ones; the
  prompt tells the model to use only what answers the question.
- Retrieval has no relevance floor, so an off-topic question still costs a
  full LLM call before it is refused.

## What's next

- Section-aware chunking.
- Per-fact citation verification (check the cited document contains the
  claim).
- A golden-answer test set: a fixed list of questions with the document and
  page that should be retrieved, so a chunking or model change can be
  measured instead of eyeballed.
- A bigger LLM when hardware allows.
