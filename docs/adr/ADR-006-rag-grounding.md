# ADR-006: RAG grounding strategy

**Status:** Accepted · 2026-10-05

## Decision
1. **Ingestion:**
   - validate the upload (type, size, page count);
   - parse and clean it (de-hyphenate, strip repeated headers and footers);
   - split it **per page**, so chunks never span pages and page citations are exact and 1-based;
   - flag instruction-like text.
2. **Retrieval:**
   - search with both the raw question and the classifier's standalone rewrite, then merge the results;
   - rerank with 0.75 × cosine + 0.25 × lexical overlap;
   - drop chunks below a cosine floor (`NOVA_MIN_RELEVANCE`).

   The floor is calibrated on a dev set that is separate from the evaluation set
   (`python -m evaluation.calibrate_threshold`).
3. **Evidence gate:** if no chunk clears the floor, Nova runs one corrective keyword retrieval. If that also
   finds nothing, it answers "I couldn't find sufficient evidence…" without calling the LLM.
4. **Generation:** sources are passed as delimited `<source id=…>` blocks marked untrusted, and the model cites
   them by id.
5. **Verification:**
   - citation ids are validated and then rendered from stored metadata;
   - every number in the answer must appear in the evidence or the calculator output;
   - a failure triggers one bounded retry, then an explicit caveat;
   - an explicit "the sources do not contain this" is accepted as a correct abstention.

## Known limitation
A cosine threshold cannot separate *in-domain* absent questions from answerable ones. For example, "dividend
per share" in an annual report that has no dividend section scores about as high as real questions. Those
cases depend on steps 4 and 5 (grounded abstention). The evaluation reports the two mechanisms separately.
