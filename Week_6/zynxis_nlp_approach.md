# NLP Sentiment Analysis on Zynxis Intern Feedback — Approach

**Author:** Syeda Mahnoor Fatima
**Week 6 — Zynxis Internship**


## Pipeline Architecture 
1. **Text ingestion** — load the 300-row feedback CSV (`feedback_text`,
   `sentiment` label, plus `category` and `week` metadata).
2. **Preprocessing** — lowercase the text, strip punctuation and numbers,
   split it into words (tokenization), remove common filler words like "the"
   and "and" (stopword removal), and reduce words to their base form, e.g.
   "loved" → "love" (lemmatization). This strips out noise so the model can
   focus on words that actually carry sentiment.
3. **TF-IDF vectorization** — convert the cleaned text into numbers. TF-IDF
   weighs a word by how distinctive it is to a document, not just how often
   it appears, so words like "frustrating" or "loved" get more weight than
   generic words like "week" or "project" that show up everywhere.
4. **Logistic Regression** — a lightweight classifier trained on those TF-IDF
   vectors to predict positive / negative / neutral.
5. **Inference** — the same preprocessing + vectorization steps are applied
   to brand-new, unseen feedback sentences, and the trained model returns a
   predicted sentiment with a confidence score per class.

## Model Choice Rationale

Logistic Regression was used as the single model, for a few reasons:

- **Interpretability** — its coefficients map directly to words/bigrams, so
  it's easy to inspect *why* something was classified a certain way, which
  matters if Zynxis staff want to trust and audit the output.
- **Efficiency on sparse text data** — TF-IDF vectors are high-dimensional
  and sparse; linear models like Logistic Regression handle that well
  without needing much data or tuning.
- **Calibrated probability outputs** — `predict_proba` gives usable
  confidence scores out of the box, which matters for the intended use case
  (flagging uncertain predictions for human review rather than trusting
  every prediction blindly).

A single model was used deliberately, rather than comparing multiple
classifiers, since the assignment scope is about demonstrating the full NLP
pipeline end-to-end rather than a model comparision.

## Results & Observations

- Accuracy: **~82%** on a held-out 20% test split (stratified).
- The model separates neutral feedback from positive/negative almost
  perfectly, but has more trouble distinguishing positive from negative
  when a comment mixes both (e.g. "Loved the resources but my mentor barely
  checked in"). This is expected and, if anything, a sign the dataset and
  evaluation are realistic — real feedback is often mixed, and a model that
  reports 95%+ accuracy on messy text like this is usually a sign of a
  dataset problem (e.g. duplicate/near-identical text leaking into both the
  training and test sets), not a genuinely strong model.
- Top TF-IDF features per class make intuitive sense: positive comments
  lean on words like "loved", "best", "supportive"; negative comments lean
  on "frustrating", "disappointing", "cancelled"; neutral comments lean on
  flat language like "fine", "alright", "average".

**For Zynxis**, this kind of pipeline could run weekly on incoming feedback
to flag consistently negative sentiment tied to a specific mentor or
category (e.g. "resources" trending negative for three weeks straight),
giving program coordinators an early signal instead of waiting for an
end-of-cohort survey. Given how often real feedback is mixed rather than
purely one-sided, predictions with low confidence (roughly below 60%) are
better routed to a human reviewer than auto-triaged.

