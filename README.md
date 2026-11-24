# PSIP Navigator

A multilingual chatbot that answers Penn international students' questions about the Penn Student Insurance Plan (PSIP, Aetna Student Health, 2026-2027): where to go for care, referrals, what things cost, prescriptions, EOBs and bills. Every answer is grounded in Aetna's plan documents and Penn's official pages, with citations.

## Quick start

```bash
cd ~/Downloads/psip-navigator
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env              # then paste ANTHROPIC_API_KEY and VOYAGE_API_KEY into .env
python scripts/07_build_index.py  # embeds the 815 chunks with Voyage (once)
streamlit run app.py              # opens the chat page in your browser
```

Voyage key: create one in the Voyage dashboard (dash.voyageai.com → API keys), or in MongoDB Atlas (Voyage is now part of MongoDB: Atlas → Model API Keys). If the key comes from Atlas, also add `VOYAGE_BASE_URL=https://ai.mongodb.com` to `.env`. Without a payment method, the free tier is heavily rate-limited and step 07 will be slow (it retries automatically). Anthropic key: console.anthropic.com. Never paste keys into chat or commit `.env`.

## Deploy (Streamlit Community Cloud, free)

1. Push this folder to a GitHub repo (`.env`, `.venv` and caches are git-ignored; the vector index in `data/index/` is included).
2. Go to share.streamlit.io, sign in with GitHub, click **Create app**, pick the repo, branch `main`, main file `app.py`.
3. **Advanced settings → Secrets**:
   ```
   ANTHROPIC_API_KEY = "sk-ant-..."
   VOYAGE_API_KEY = "pa-..."
   ```
4. Deploy. You get a `https://<name>.streamlit.app` link; pushing to GitHub redeploys automatically.

Usage limits (so a public link can't run up the API bill): 20 questions per visitor per hour and 200 per hour for all visitors together, questions up to 1,000 characters. Change with the `PER_SESSION_PER_HOUR` / `ALL_USERS_PER_HOUR` secrets. Free apps sleep after 12 hours without visits: open the link once before a demo.

## How a message is answered

```
student message (any language)
  1. Safety router      fixed rules (English + Chinese keywords): crisis or emergency -> fixed contacts, stop.  No AI.
  2. Rewrite            Claude Haiku: detects the language, writes 1-3 English search queries in plan terms
                        ("blood test" -> "outpatient lab services"), pulls out drug names, flags crisis, and judges
                        whether the message is in scope. Off-topic or small talk -> short reply in the student's
                        language, no search, no Voyage, no answer model (when unsure it counts as in scope).
  3. Hybrid search      BM25 keyword search on the English queries
                        + Voyage vector search on the student's original question, only for non-English questions
                          (VECTOR_MODE=auto; one short call per message, cached; if Voyage is rate-limited the
                          bot carries on with BM25 and says so in the status line)
                        -> merged with reciprocal rank fusion -> top 8 chunks.
  4. Answer             Claude Sonnet answers in the student's language from those chunks only, keeps terms in English
                        ("免赔额 (deductible)"), cites chunk IDs, and calls two tools for every number:
                          calculate_cost  deductible -> copay -> coinsurance, referral rule, out-of-pocket cap
                          lookup_drug     tier, PA/QL/ST, copay from the 2026 drug guide (+ covered generic versions)
  5. Sources            only IDs the answer cited and that were actually retrieved are listed.
```

This matches the proposal's split: AI for understanding everyday wording (R1), searching across documents (R2) and answering in the student's language (R3); fixed code for costs (R4), emergencies (R5) and hand-offs to people (R6).

## Folder layout

```
app.py                Streamlit chat page
psip_navigator/       config, router, search, tools (calculator + drug lookup), llm (rewrite + answer), bot (pipeline)
scripts/01-06         data preprocessing (already run; outputs in data/)
scripts/07            build the Voyage vector index
scripts/08            translate the test questions (Chinese by default) for the multilingual eval
scripts/09            evaluation
tests/                offline tests with fake Claude/Voyage (no keys needed): python -m pytest tests -q
raw/                  source PDFs, captured web pages, the team's test set
data/                 processed data (below)
results/              eval outputs
```

## Data in `data/`

| File | What it is |
| --- | --- |
| `corpus.jsonl` | **What the chatbot searches:** 815 chunks = 729 official + 86 team-verified answers |
| `chunks_pdf.jsonl` | 521 chunks from the 5 Aetna plan PDFs, cut at section headings |
| `chunks_rx.jsonl` | 12 chunks from the drug guide's explanation pages plus a tier-code legend |
| `chunks_web.jsonl` | 196 chunks from 47 web pages (Wellness at Penn, Penn Global, Penn Medicine, HealthCare.gov glossary) |
| `chunks_curated.jsonl` | 86 test-set answers marked Ready, each citing the official codes it came from |
| `formulary.jsonl` | 4,233 drug listings (name, generic/brand, tier, PA/QL/ST, class, page), looked up by name |
| `plan_costs.json` | Deductibles, out-of-pocket limits, copays, coinsurance, drug tiers; every number has a checked quote |
| `eval_set.jsonl` | The 106 test questions with expected answers and sources |
| `data_report.md` | Data checks (schema, sizes, cost quotes, key facts) |

Chunk fields: `id`, `source_code`, `family`, `source_title`, `source_type` (official / curated), `url`, `heading_path`, `text` (original wording; PDF table columns separated by ` | `), `page_start`, `page_end`, `plan_year`, `date_captured`, `cites`.

## Evaluation

```bash
python scripts/09_eval.py --level retrieval --lang en                 # did search find an expected source? (hit@1/3/8)
python scripts/08_translate_eval.py                                   # make the Chinese test set (once)
python scripts/09_eval.py --level retrieval --lang zh                 # same questions in Chinese
python scripts/09_eval.py --level answer --lang en --limit 20         # full answers graded by a Claude judge
python scripts/09_eval.py --level answer --lang zh --limit 20
```

The team's answers are left out by default so a test question can't find its own answer; add `--include-curated` to test the full system. Only questions marked Ready are scored (`--status ""` for all).

**Retrieval results** (93 Ready questions, team answers excluded; an expected source in the top k):

| Setup | hit@1 | hit@3 | hit@8 |
| --- | --- | --- | --- |
| BM25 on the raw English question (no AI) | 26% | 55% | 74% |
| Full system, English (rewrite + BM25; vectors skipped in `auto` mode) | 40% | 65% | 83% |
| Full system, Chinese (rewrite + BM25 + Voyage vectors) | 46% | 67% | 86% |

**Answer results** (30 "before care" questions per language, graded by a Claude judge against the team's reference answers, after pinning the core plan rules):

| Language | Correct | Partial | Wrong | Right language |
| --- | --- | --- | --- | --- |
| English | 17 | 13 | 0 | 30/30 |
| Chinese | 21 | 8 | 1 | 29/29 reported |

The one wrong answer (Q001) gave the Counseling address instead of Medical Care; the locations chunk is now always included and the prompt says which office to give.

Most remaining misses are labeling, not search: the EOB and term-definition questions (Q51-56, Q88, Q89, Q91) expect the HealthCare.gov glossary while the bot retrieves the policy's own definitions or Penn Medicine's billing FAQ, and hand-off questions (Q100-102) correctly need no document. No test question was wrongly judged off-topic. To test vectors for English too: `VECTOR_MODE=always python scripts/09_eval.py --level retrieval --lang en`.

## Rebuilding the data

```bash
brew install poppler            # pdftotext, used by scripts 01 and 05
python scripts/01_parse_pdfs.py
python scripts/02_chunk_web.py
python scripts/03_build_testset.py
python scripts/05_parse_formulary.py
python scripts/04_check_data.py
python scripts/06_build_corpus.py
python scripts/07_build_index.py   # always re-embed after the corpus changes
```

## Open items

1. **Test-set answers may be missing the deductible.** The plan applies the deductible first, then the copay (AET-POL-005); the SBC says its copays are "after your deductible has been met, if a deductible applies." It is waived for the ER, prescriptions, mental health, preventive care and vision, not for office visits, urgent care or lab work. So "urgent care is a $35 copay" holds only once the $400 in-network deductible is met. Check Q24, Q25, Q27, Q28, Q43, Q58 with the Insurance Navigators. The calculator follows the plan documents.
2. **EOB questions (Q51-Q56) cite the HealthCare.gov glossary**, which has no EOB entry; the official EOB text is in AET-POL-274 and Penn Medicine's billing FAQ (PMD-BFQ-13 to 15). Retrieval eval counts these as misses until the test set's sources are updated.
3. **Section labels:** the test set's AET-RXP, PNN, FRM, TRV and SUR are sections of the PDFs; script 03 maps them (`ALIASES`). Two facts (68,000 pharmacies, On Call International) exist only in the team's answers.
4. **Drug guide quirks:** some drugs appear twice with different tiers (e.g. a sertraline strength listed as both NF and PG); the bot shows every match and says to confirm with the pharmacy or Aetna.
5. **Safety router keywords** are English and Chinese only; add other languages' keywords in `psip_navigator/router.py`. The rewrite step's crisis flag is a second check for any language.
