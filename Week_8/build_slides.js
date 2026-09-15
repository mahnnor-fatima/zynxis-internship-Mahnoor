const pptxgen = require("pptxgenjs");

// ---- Palette: Midnight Executive ----
const NAVY = "1E2761";
const ICE = "CADCFC";
const WHITE = "FFFFFF";
const ACCENT = "F2A65A"; // warm amber accent for contrast against navy
const DARK_TEXT = "1E2761";
const MUTED = "6B7280";
const GOOD = "27AE60";
const BAD = "C0392B";

const FONT_HEAD = "Cambria";
const FONT_BODY = "Calibri";

function newDeck() {
  const p = new pptxgen();
  p.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
  return p;
}

function titleBar(slide, text, opts = {}) {
  slide.addText(text, {
    x: 0.5, y: 0.4, w: 12.3, h: 0.9,
    fontFace: FONT_HEAD, fontSize: 30, bold: true, color: opts.color || NAVY,
    align: "left", isTextBox: true, margin: 0,
  });
}

function pageNum(slide, n) {
  slide.addText(String(n), {
    x: 12.6, y: 7.05, w: 0.5, h: 0.3, fontFace: FONT_BODY, fontSize: 10,
    color: MUTED, align: "right", isTextBox: true, margin: 0,
  });
}

const deck = newDeck();

// ---------------- SLIDE 1: TITLE ----------------
{
  const s = deck.addSlide();
  s.background = { color: NAVY };
  s.addText("ZYNXIS FINAL CAPSTONE", {
    x: 0.9, y: 2.55, w: 11.5, h: 0.5, fontFace: FONT_BODY, fontSize: 16, bold: true,
    color: ACCENT, charSpacing: 3, isTextBox: true, margin: 0,
  });
  s.addText("Intern Performance Predictor", {
    x: 0.9, y: 3.05, w: 11.5, h: 1.3, fontFace: FONT_HEAD, fontSize: 44, bold: true,
    color: WHITE, isTextBox: true, margin: 0,
  });
  s.addText("An end-to-end machine learning system for predicting intern success", {
    x: 0.9, y: 4.55, w: 10.5, h: 0.6, fontFace: FONT_BODY, fontSize: 18, italic: true,
    color: ICE, isTextBox: true, margin: 0,
  });
  s.addText("Mahnoor  \u2022  Zynxis Internship Program  \u2022  Week 8", {
    x: 0.9, y: 6.6, w: 10.5, h: 0.4, fontFace: FONT_BODY, fontSize: 13,
    color: ICE, isTextBox: true, margin: 0,
  });
}

// ---------------- SLIDE 2: PROBLEM ----------------
{
  const s = deck.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "The Problem");
  s.addText(
    "Zynxis tracks rich performance data on every intern \u2014 but deciding who's on track " +
    "to succeed still relies on manual review.",
    { x: 0.5, y: 1.35, w: 6.2, h: 1.6, fontFace: FONT_BODY, fontSize: 16, color: DARK_TEXT, isTextBox: true, margin: 0 }
  );
  s.addText(
    "Goal: predict, from measurable metrics, whether an intern will become a high performer " +
    "\u2014 giving coordinators an early, data-driven signal to guide mentorship.",
    { x: 0.5, y: 3.1, w: 6.2, h: 1.6, fontFace: FONT_BODY, fontSize: 16, bold: true, color: NAVY, isTextBox: true, margin: 0 }
  );

  // Right side: stat callout card
  s.addShape("roundRect", {
    x: 7.2, y: 1.35, w: 5.6, h: 4.9, rectRadius: 0.12,
    fill: { color: "F4F6FB" }, line: { type: "none" },
  });
  s.addText("What we track per intern", {
    x: 7.6, y: 1.65, w: 4.8, h: 0.4, fontFace: FONT_HEAD, fontSize: 16, bold: true, color: NAVY, isTextBox: true, margin: 0,
  });
  const items = [
    "Technical Score", "Project Completion Rate", "Attendance & Punctuality",
    "Soft Skills Rating", "Code Review Score", "Prior Experience", "Education Level",
  ];
  let y = 2.25;
  items.forEach((it) => {
    s.addShape("ellipse", { x: 7.6, y: y + 0.08, w: 0.12, h: 0.12, fill: { color: ACCENT }, line: { type: "none" } });
    s.addText(it, { x: 7.85, y: y - 0.03, w: 4.6, h: 0.35, fontFace: FONT_BODY, fontSize: 14, color: DARK_TEXT, isTextBox: true, margin: 0 });
    y += 0.52;
  });
  pageNum(s, 2);
}

// ---------------- SLIDE 3: DATA ----------------
{
  const s = deck.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "Data Collection & Preparation");

  const stats = [
    { n: "500", l: "Intern records" },
    { n: "7", l: "Input features" },
    { n: "0", l: "Missing values" },
    { n: "62/38", l: "Class balance" },
  ];
  let x = 0.5;
  const cardW = 1.5;
  stats.forEach((st) => {
    s.addShape("roundRect", { x, y: 1.4, w: cardW, h: 1.35, rectRadius: 0.09, fill: { color: NAVY }, line: { type: "none" } });
    s.addText(st.n, { x, y: 1.52, w: cardW, h: 0.65, align: "center", fontFace: FONT_HEAD, fontSize: 22, bold: true, color: WHITE, isTextBox: true, margin: 0 });
    s.addText(st.l, { x: x - 0.05, y: 2.18, w: cardW + 0.1, h: 0.5, align: "center", fontFace: FONT_BODY, fontSize: 10, color: ICE, isTextBox: true, margin: 0 });
    x += cardW + 0.2;
  });

  s.addText("Preprocessing steps", {
    x: 0.5, y: 3.25, w: 6, h: 0.4, fontFace: FONT_HEAD, fontSize: 16, bold: true, color: NAVY, isTextBox: true, margin: 0,
  });
  const steps = [
    "One-hot encode Education_Level (drop_first)",
    "Standardize 6 numeric features (fit on train only)",
    "80/20 stratified train/test split (seed=42)",
  ];
  let sy = 3.75;
  steps.forEach((st, i) => {
    s.addText(`${i + 1}`, { x: 0.5, y: sy, w: 0.4, h: 0.4, fontFace: FONT_BODY, fontSize: 14, bold: true, color: ACCENT, isTextBox: true, margin: 0 });
    s.addText(st, { x: 0.95, y: sy, w: 5.6, h: 0.4, fontFace: FONT_BODY, fontSize: 14, color: DARK_TEXT, isTextBox: true, margin: 0 });
    sy += 0.55;
  });

  s.addImage({ path: "assets_correlation_heatmap.png", x: 7.1, y: 1.4, w: 5.7, h: 4.9 });
  pageNum(s, 3);
}

// ---------------- SLIDE 4: MODEL ----------------
{
  const s = deck.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "Model Training");

  s.addShape("roundRect", { x: 0.5, y: 1.4, w: 5.9, h: 4.9, rectRadius: 0.12, fill: { color: "F4F6FB" }, line: { type: "none" } });
  s.addText("Random Forest Classifier", {
    x: 0.9, y: 1.7, w: 5.1, h: 0.5, fontFace: FONT_HEAD, fontSize: 20, bold: true, color: NAVY, isTextBox: true, margin: 0,
  });
  s.addText(
    "Chosen for its ability to capture non-linear interactions between features and its " +
    "built-in, interpretable feature importance scores.",
    { x: 0.9, y: 2.3, w: 5.1, h: 1.0, fontFace: FONT_BODY, fontSize: 14, color: DARK_TEXT, isTextBox: true, margin: 0 }
  );
  const hp = [
    ["n_estimators", "100"], ["max_depth", "7"],
    ["min_samples_split", "4"], ["random_state", "42"],
  ];
  let hy = 3.55;
  hp.forEach(([k, v]) => {
    s.addText(k, { x: 0.9, y: hy, w: 3.2, h: 0.4, fontFace: FONT_BODY, fontSize: 14, color: MUTED, isTextBox: true, margin: 0 });
    s.addText(v, { x: 4.1, y: hy, w: 1.9, h: 0.4, fontFace: FONT_BODY, fontSize: 14, bold: true, color: NAVY, isTextBox: true, margin: 0 });
    hy += 0.5;
  });

  s.addShape("roundRect", { x: 6.65, y: 1.4, w: 6.15, h: 4.9, rectRadius: 0.12, fill: { color: NAVY }, line: { type: "none" } });
  s.addText("Also evaluated", {
    x: 7.05, y: 1.7, w: 5.4, h: 0.4, fontFace: FONT_HEAD, fontSize: 16, bold: true, color: WHITE, isTextBox: true, margin: 0,
  });
  ["Logistic Regression \u2014 baseline linear model", "Single Decision Tree \u2014 prone to overfitting"].forEach((t, i) => {
    s.addText(t, { x: 7.05, y: 2.25 + i * 0.55, w: 5.4, h: 0.5, fontFace: FONT_BODY, fontSize: 14, color: ICE, isTextBox: true, margin: 0 });
  });
  s.addText("Random Forest won on both accuracy and interpretability, so it's the model deployed in the final app.", {
    x: 7.05, y: 3.6, w: 5.4, h: 1.4, fontFace: FONT_BODY, fontSize: 14, italic: true, color: WHITE, isTextBox: true, margin: 0,
  });
  pageNum(s, 4);
}

// ---------------- SLIDE 5: EVALUATION METRICS ----------------
{
  const s = deck.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "Evaluation Results");

  s.addChart(
    "bar",
    [{
      name: "Test Set Score",
      labels: ["Accuracy", "Precision", "Recall", "F1"],
      values: [0.69, 0.746, 0.758, 0.752],
    }],
    {
      x: 0.5, y: 1.4, w: 6.3, h: 4.9,
      showTitle: true, title: "Test Set Metrics", titleFontSize: 14, titleColor: NAVY,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0%",
      chartColors: [NAVY],
      catAxisLabelColor: DARK_TEXT, valAxisLabelColor: DARK_TEXT,
      valAxisMaxVal: 1, valGridLine: { color: "E5E7EB", size: 1 },
      catGridLine: { style: "none" }, showLegend: false,
      dataLabelColor: NAVY, dataLabelFontSize: 12,
    }
  );

  s.addImage({ path: "assets_roc_curve.png", x: 7.0, y: 1.4, w: 5.8, h: 4.6 });
  s.addText("ROC-AUC = 0.71 \u2014 well above the 0.50 random baseline", {
    x: 7.0, y: 6.05, w: 5.8, h: 0.3, align: "center", fontFace: FONT_BODY, fontSize: 12, italic: true, color: MUTED, isTextBox: true, margin: 0,
  });
  pageNum(s, 5);
}

// ---------------- SLIDE 6: FEATURE IMPORTANCE ----------------
{
  const s = deck.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "What Drives the Prediction?");
  s.addImage({ path: "assets_feature_importance.png", x: 0.5, y: 1.35, w: 7.6, h: 4.9 });

  s.addShape("roundRect", { x: 8.4, y: 1.35, w: 4.4, h: 4.9, rectRadius: 0.12, fill: { color: "F4F6FB" }, line: { type: "none" } });
  s.addText("Key takeaway", {
    x: 8.8, y: 1.65, w: 3.6, h: 0.4, fontFace: FONT_HEAD, fontSize: 16, bold: true, color: NAVY, isTextBox: true, margin: 0,
  });
  s.addText(
    "Technical Score and Code Review Score together drive roughly half of the model's decisions.",
    { x: 8.8, y: 2.2, w: 3.6, h: 1.1, fontFace: FONT_BODY, fontSize: 14, color: DARK_TEXT, isTextBox: true, margin: 0 }
  );
  s.addText(
    "Education Level barely matters \u2014 the model rewards demonstrated performance over credentials, a fair outcome for an evaluation tool.",
    { x: 8.8, y: 3.5, w: 3.6, h: 1.6, fontFace: FONT_BODY, fontSize: 14, italic: true, color: NAVY, isTextBox: true, margin: 0 }
  );
  pageNum(s, 6);
}

// ---------------- SLIDE 7: DEPLOYED APP ----------------
{
  const s = deck.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "Deployed Interface");
  s.addText(
    "A Streamlit web app lets a coordinator enter an intern's metrics and get an instant, " +
    "explainable prediction.",
    { x: 0.5, y: 1.3, w: 5.9, h: 1.0, fontFace: FONT_BODY, fontSize: 16, color: DARK_TEXT, isTextBox: true, margin: 0 }
  );
  const feats = [
    "Predict: verdict + probability + plain-language \u201cwhy\u201d",
    "Batch Prediction: upload a CSV, screen a whole cohort",
    "Model Comparison: RF vs. Logistic Regression vs. Tree",
    "Feature Importance: interactive chart of what matters",
  ];
  let fy = 2.4;
  feats.forEach((t) => {
    s.addShape("ellipse", { x: 0.5, y: fy + 0.08, w: 0.14, h: 0.14, fill: { color: ACCENT }, line: { type: "none" } });
    s.addText(t, { x: 0.85, y: fy - 0.05, w: 5.5, h: 0.45, fontFace: FONT_BODY, fontSize: 14, color: DARK_TEXT, isTextBox: true, margin: 0 });
    fy += 0.6;
  });
  s.addText("streamlit run app.py \u2014 runs locally or on Streamlit Community Cloud", {
    x: 0.5, y: 5.3, w: 5.9, h: 0.5, fontFace: "Courier New", fontSize: 12, color: NAVY, isTextBox: true, margin: 0,
  });

  // Right: mock app preview card
  s.addShape("roundRect", { x: 7.0, y: 1.3, w: 5.8, h: 5.5, rectRadius: 0.12, fill: { color: NAVY }, line: { type: "none" } });
  s.addText("\uD83D\uDCCA Zynxis Intern Performance Predictor", {
    x: 7.35, y: 1.55, w: 5.1, h: 0.5, fontFace: FONT_BODY, fontSize: 14, bold: true, color: WHITE, isTextBox: true, margin: 0,
  });
  s.addShape("roundRect", { x: 7.35, y: 2.25, w: 5.1, h: 0.9, rectRadius: 0.08, fill: { color: "27AE60" }, line: { type: "none" } });
  s.addText("\u2705 High Performer / Likely to be Placed", {
    x: 7.55, y: 2.4, w: 4.7, h: 0.6, fontFace: FONT_BODY, fontSize: 13, bold: true, color: WHITE, isTextBox: true, margin: 0,
  });
  s.addText("Predicted Probability", { x: 7.35, y: 3.35, w: 5.1, h: 0.35, fontFace: FONT_BODY, fontSize: 12, color: ICE, isTextBox: true, margin: 0 });
  s.addText("91%", { x: 7.35, y: 3.65, w: 5.1, h: 0.6, fontFace: FONT_HEAD, fontSize: 30, bold: true, color: WHITE, isTextBox: true, margin: 0 });
  s.addShape("roundRect", { x: 7.35, y: 4.35, w: 5.1, h: 0.18, rectRadius: 0.09, fill: { color: "3B4A8C" }, line: { type: "none" } });
  s.addShape("roundRect", { x: 7.35, y: 4.35, w: 4.65, h: 0.18, rectRadius: 0.09, fill: { color: ACCENT }, line: { type: "none" } });
  pageNum(s, 7);
}

// ---------------- SLIDE 8: LIMITATIONS & FUTURE WORK ----------------
{
  const s = deck.addSlide();
  s.background = { color: WHITE };
  titleBar(s, "Limitations & Future Work");

  s.addShape("roundRect", { x: 0.5, y: 1.4, w: 5.9, h: 4.8, rectRadius: 0.12, fill: { color: "FDEEEA" }, line: { type: "none" } });
  s.addText("Current limitations", { x: 0.9, y: 1.65, w: 5.1, h: 0.4, fontFace: FONT_HEAD, fontSize: 16, bold: true, color: BAD, isTextBox: true, margin: 0 });
  [
    "Modest dataset size (500 records)",
    "69% accuracy \u2014 supports, doesn't replace, human judgment",
    "Weaker recall (58%) on the \u201cNeeds Support\u201d class",
  ].forEach((t, i) => {
    s.addText("\u2022 " + t, { x: 0.9, y: 2.2 + i * 0.65, w: 5.1, h: 0.6, fontFace: FONT_BODY, fontSize: 14, color: DARK_TEXT, isTextBox: true, margin: 0 });
  });

  s.addShape("roundRect", { x: 6.9, y: 1.4, w: 5.9, h: 4.8, rectRadius: 0.12, fill: { color: "E8F8F0" }, line: { type: "none" } });
  s.addText("Planned improvements", { x: 7.3, y: 1.65, w: 5.1, h: 0.4, fontFace: FONT_HEAD, fontSize: 16, bold: true, color: GOOD, isTextBox: true, margin: 0 });
  [
    "Larger dataset + class-weighting or SMOTE",
    "SHAP-based explanations per prediction",
    "Batch CSV upload to screen a full cohort",
  ].forEach((t, i) => {
    s.addText("\u2022 " + t, { x: 7.3, y: 2.2 + i * 0.65, w: 5.1, h: 0.6, fontFace: FONT_BODY, fontSize: 14, color: DARK_TEXT, isTextBox: true, margin: 0 });
  });
  pageNum(s, 8);
}

// ---------------- SLIDE 9: CLOSING ----------------
{
  const s = deck.addSlide();
  s.background = { color: NAVY };
  s.addText("Thank You", {
    x: 0.9, y: 2.9, w: 11.5, h: 1.0, fontFace: FONT_HEAD, fontSize: 40, bold: true, color: WHITE, isTextBox: true, margin: 0,
  });
  s.addText("A complete pipeline \u2014 data, model, evaluation, and a deployed app \u2014 giving Zynxis a data-backed signal for intern success.", {
    x: 0.9, y: 3.85, w: 10.8, h: 0.8, fontFace: FONT_BODY, fontSize: 16, italic: true, color: ICE, isTextBox: true, margin: 0,
  });
  s.addText("Questions?", {
    x: 0.9, y: 5.9, w: 6, h: 0.5, fontFace: FONT_BODY, fontSize: 15, color: ACCENT, isTextBox: true, margin: 0,
  });
}

deck.writeFile({ fileName: "Zynxis_Capstone_Slides.pptx" }).then(() => console.log("Slides written."));
