# Week 1 · The whole pipeline, at toy scale

About forty web pages in, a very small chat model out. Every stage a real LLM goes through, shrunk to
run on one machine in an afternoon. The model will be bad. The point is to build each stage once, so
the later projects each have a place on the map.

Written up at [danielrushton.com/writing/f1-the-whole-pipeline](https://danielrushton.com/writing/f1-the-whole-pipeline/).

| Stage | Script | What it does | Real-scale equivalent |
| --- | --- | --- | --- |
| 1 · Seed URLs | `seed_urls.tsv` | ~40 hand-picked pages, each labelled `keep` or `drop` | A Common Crawl snapshot |
| 2 · Fetch | `fetch.py` | Polite download: User-Agent, delay, robots.txt | Common Crawl's crawler |
| 3 · Extract | `extract_and_filter.py` | Main text only, via trafilatura | FineWeb's trafilatura pass |
| 4 · Filter + dedup | `extract_and_filter.py` | Language ID, Gopher repetition and quality, C4 rules, MinHash dedup, with datatrove | FineWeb, with the same library |
| 5 · Score the filters | `score_filters.py` | What survived vs. the labels: a first, tiny eval | Data ablations |
| 6 · Tokenize | `train_tokenizer.py` | Byte-level BPE, 4,096 tokens | GPT-style BPE, 50k–200k tokens |
| 7 · Pretrain | `pretrain.py` | A ~1M-parameter GPT, next-token prediction | Billions of parameters, trillions of tokens |
| 8 · Fine-tune for chat | `sft.py` | 500 conversations from smol-smoltalk, loss on the assistant's turns only | Supervised fine-tuning, then preference tuning and RL |
| 9 · Chat | `chat.py` | Ask a question, get an answer | A deployed assistant |

## Run it

Python 3.10 or later. A GPU helps but isn't needed: the defaults train in minutes on a laptop CPU.

Set up this project's own virtual environment from the repo root, then work inside it:

```bash
./setup_venv.sh week-1-mini-pipeline     # creates week-1-mini-pipeline/.venv and installs requirements.txt
source week-1-mini-pipeline/.venv/bin/activate
cd week-1-mini-pipeline
```

Then run the stages in order:

```bash
python fetch.py seed_urls.tsv     # stage 2: raw pages into data/raw/
python extract_and_filter.py      # stages 3–4: data/clean/, and data/excluded/<filter>/ for what each dropped
python score_filters.py           # stage 5: kept vs. expected, page by page
python train_tokenizer.py         # stage 6: out/tokenizer.json
python pretrain.py                # stage 7: out/base.pt
python sft.py                     # stage 8: out/chat.pt (downloads part of smol-smoltalk)
python chat.py                    # stage 9
```

The first `extract_and_filter.py` run downloads fastText's language-ID model (about 130 MB). Use
`--skip-language-filter` to run without it. `sft.py --data my_conversations.jsonl` uses a local file
of `{"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}` lines
instead of downloading smol-smoltalk.

Each stage overwrites its own outputs, so any stage can be rerun on its own.

## The seed list

Four groups, chosen so that every filter has something to catch:

- **clean** (21): Simple English Wikipedia articles on physics, consciousness, computing and music.
- **book** (6): Project Gutenberg plain text. *Pride and Prejudice* is over 100,000 words, past Gopher's
  default length limit, so expect the filter to drop a page it shouldn't.
- **duplicate** (5): an exact copy of a book, the same book as HTML, and mobile versions of three
  articles. MinHash should keep one of each.
- **junk** (8): index, category, login and log pages, plus German and French articles for the language
  filter. Some are blocked by robots.txt, which counts as dropped at the fetch stage.

Before relying on the list, run `fetch.py` and check that every URL loads. Wikimedia asks for a
descriptive User-Agent with a way to contact you: set `FETCH_USER_AGENT` or pass `--user-agent`.

## What to look at

- `score_filters.py`: which good pages each filter lost, and which bad pages got through. Every filter
  is a classifier with false positives.
- `train_tokenizer.py`: how many characters fit in a token, and how common words split.
- `pretrain.py`: the gap between train and validation loss. With a few hundred thousand tokens the
  model memorizes; that gap is what more data closes.
- `chat.py`: fluent-looking nonsense, in the right format. The format is what SFT taught it.

## Credits

The model in `common.py` follows Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).
[nanochat](https://github.com/karpathy/nanochat) is the same pipeline at real scale. The filters and
deduplication are Hugging Face's [datatrove](https://github.com/huggingface/datatrove), the library
behind [FineWeb](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/). Conversations come from
[smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) (Apache-2.0).
