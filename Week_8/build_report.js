const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, PageBreak,
  ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, BorderStyle, Header, Footer, PageNumber
} = require("docx");
const fs = require("fs");

const PAGE = { width: 12240, height: 15840 }; // US Letter, DXA

function img(path, widthPx, heightPx) {
  return new ImageRun({
    type: "png",
    data: fs.readFileSync(path),
    transformation: { width: widthPx, height: heightPx },
  });
}

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } });
}
function body(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    spacing: { after: 120 },
  });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
}

// Metrics table
function metricsTable(customRows) {
  const rows = customRows || [
    ["Metric", "Score"],
    ["Accuracy", "69.0%"],
    ["Precision", "74.6%"],
    ["Recall", "75.8%"],
    ["F1 Score", "0.752"],
    ["ROC-AUC", "0.710"],
  ];
  const nCols = rows[0].length;
  const colWidths = nCols === 2 ? [6000, 3000] : [3600, 1800, 1800, 1800];
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: rows.map((r, i) => new TableRow({
      children: r.map((cellText, ci) => new TableCell({
        width: { size: colWidths[ci], type: WidthType.DXA },
        shading: i === 0 ? { type: ShadingType.CLEAR, fill: "2C3E50" } : undefined,
        children: [new Paragraph({
          children: [new TextRun({ text: cellText, bold: i === 0, color: i === 0 ? "FFFFFF" : "000000" })],
        })],
      })),
    })),
  });
}

const doc = new Document({
  sections: [
    // TITLE PAGE
    {
      properties: { page: { size: PAGE } },
      children: [
        new Paragraph({ text: "", spacing: { before: 2000 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Zynxis Intern Performance Predictor", bold: true, size: 44 })],
          spacing: { after: 200 },
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "An End-to-End Machine Learning System for Predicting Intern Success", size: 26, italics: true })],
          spacing: { after: 600 },
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Final Capstone Project \u2014 Week 8", size: 24 })],
          spacing: { after: 100 },
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Prepared by Mahnoor", size: 24 })],
          spacing: { after: 100 },
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Zynxis Internship Program", size: 24 })],
        }),
        new Paragraph({ children: [new PageBreak()] }),
      ],
    },
    // MAIN CONTENT
    {
      properties: { page: { size: PAGE } },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            children: [new TextRun({ text: "Zynxis Intern Performance Predictor \u2014 Capstone Report", size: 16, color: "808080" })],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })],
          })],
        }),
      },
      children: [
        h1("1. Executive Summary"),
        body(
          "This project delivers an end-to-end machine learning system that predicts whether a Zynxis " +
          "intern is likely to become a high performer and be placed, based on measurable performance " +
          "signals collected during the internship. The system covers the full pipeline: data preparation, " +
          "model training and evaluation, and a deployed Streamlit web application that lets a program " +
          "coordinator enter an intern's metrics and get an instant prediction with a confidence score " +
          "and a plain-language explanation \u2014 or screen a whole cohort at once via batch CSV upload."
        ),
        body(
          "The final model is a Random Forest Classifier trained on 500 intern records, achieving 69% " +
          "accuracy and an F1 score of 0.75 on held-out test data. The most influential factors in the " +
          "prediction are Technical Score and Code Review Score, followed by Soft Skills Rating and " +
          "Project Completion Rate \u2014 a result that aligns with intuitive expectations about what drives " +
          "internship success."
        ),

        h1("2. Problem Statement"),
        body(
          "Zynxis runs a structured internship program and tracks a range of performance metrics for " +
          "each intern \u2014 technical assessments, project delivery, attendance, soft skills, and code " +
          "review feedback. Today, deciding which interns are on track to become high performers (and " +
          "which need additional support) relies on manual review of these metrics by program staff."
        ),
        body(
          "The goal of this project is to automate that judgment with a machine learning model: given " +
          "an intern's metrics at any point in the program, predict the probability that they will end " +
          "up classified as a high performer / placement-ready candidate. This gives coordinators an " +
          "early, data-driven signal to identify interns who may need extra mentorship, without replacing " +
          "human judgment."
        ),

        h1("3. Data Collection & Preparation"),
        h2("3.1 Data Source"),
        body(
          "The dataset (zynxis_intern_performance.csv) contains 500 intern records, each with 7 input " +
          "features and one binary outcome label (Placed_Or_HighPerformer)."
        ),
        bullet("Technical_Score (0\u2013100) \u2014 performance on technical coding assessments"),
        bullet("Project_Completion_Rate (%) \u2014 share of assigned project work completed"),
        bullet("Attendance_Punctuality (%) \u2014 attendance and on-time record"),
        bullet("Soft_Skills_Rating (1\u20135) \u2014 communication and collaboration rating"),
        bullet("Code_Review_Score (1\u201310) \u2014 average code review feedback score"),
        bullet("Prior_Experience_Months (0\u201324) \u2014 relevant experience before the internship"),
        bullet("Education_Level \u2014 High_School / Undergrad / Grad"),

        h2("3.2 Data Quality"),
        body(
          "The dataset was checked for missing values and duplicate records \u2014 none were found. " +
          "The target class is moderately imbalanced: 62% of interns are labeled high performers versus " +
          "38% needing support, which was accounted for by using stratified train/test splitting."
        ),
        new Paragraph({
          children: [img("assets_class_balance.png", 260, 210)],
          alignment: AlignmentType.CENTER,
          spacing: { before: 100, after: 200 },
        }),

        h2("3.3 Exploratory Data Analysis"),
        body(
          "Feature distributions were examined to understand the shape and spread of each metric, and a " +
          "correlation matrix was used to check for multicollinearity and to see which raw features " +
          "correlate most with the outcome."
        ),
        new Paragraph({
          children: [img("assets_feature_distributions.png", 470, 253)],
          alignment: AlignmentType.CENTER,
          spacing: { before: 100, after: 200 },
        }),
        new Paragraph({
          children: [img("assets_correlation_heatmap.png", 340, 291)],
          alignment: AlignmentType.CENTER,
          spacing: { before: 100, after: 200 },
        }),
        body(
          "Technical_Score and Code_Review_Score show the strongest positive correlation with the outcome, " +
          "which foreshadows their importance in the trained model (Section 5.3)."
        ),

        h2("3.4 Preprocessing"),
        bullet("Categorical encoding: Education_Level one-hot encoded (drop_first=True), yielding Education_Level_High_School and Education_Level_Undergrad as binary columns, with Grad as the reference category."),
        bullet("Feature scaling: all six numeric features standardized with scikit-learn's StandardScaler (fit on the training split only, to avoid data leakage)."),
        bullet("Train/test split: 80/20 stratified split (random_state=42) to preserve class balance in both sets."),

        new Paragraph({ children: [new PageBreak()] }),

        h1("4. Model Training"),
        h2("4.1 Algorithm Selection"),
        body(
          "A Random Forest Classifier was selected as the final model after comparing it against two " +
          "baselines \u2014 Logistic Regression and a single Decision Tree \u2014 trained on the identical " +
          "80/20 split for a fair comparison. Random Forest was chosen for its ability to capture " +
          "non-linear interactions between features (e.g. technical skill compensating for lower " +
          "attendance) without heavy manual feature engineering, and for its built-in feature importance " +
          "scores, which make the model's reasoning interpretable to non-technical stakeholders. All " +
          "three models are also available side by side in the deployed app's Model Comparison tab " +
          "(Section 6.3)."
        ),
        metricsTable([
          ["Model", "Accuracy", "F1", "ROC-AUC"],
          ["Random Forest (deployed)", "69.0%", "0.752", "0.710"],
          ["Logistic Regression", "68.0%", "0.733", "0.715"],
          ["Decision Tree", "65.0%", "0.701", "0.660"],
        ]),
        new Paragraph({ text: "", spacing: { after: 100 } }),
        body(
          "Random Forest and Logistic Regression perform similarly on this dataset, with Random Forest " +
          "slightly ahead on accuracy and F1; the Decision Tree trails both, consistent with it being " +
          "more prone to overfitting on a dataset this size. Random Forest was chosen for deployment for " +
          "its combination of competitive accuracy and built-in feature importance."
        ),

        h2("4.2 Hyperparameters"),
        bullet("n_estimators = 100"),
        bullet("max_depth = 7"),
        bullet("min_samples_split = 4"),
        bullet("random_state = 42 (for reproducibility)"),

        h1("5. Evaluation"),
        h2("5.1 Test Set Metrics"),
        body("The model was evaluated on the 100 held-out test records (20% split), which it never saw during training."),
        metricsTable(),
        new Paragraph({ text: "", spacing: { after: 200 } }),

        h2("5.2 Confusion Matrix & ROC Curve"),
        new Paragraph({
          children: [img("assets_confusion_matrix.png", 250, 225)],
          alignment: AlignmentType.CENTER,
          spacing: { before: 100, after: 100 },
        }),
        new Paragraph({
          children: [img("assets_roc_curve.png", 250, 225)],
          alignment: AlignmentType.CENTER,
          spacing: { before: 100, after: 200 },
        }),
        body(
          "The model performs noticeably better at identifying high performers (recall 76%) than at " +
          "flagging interns needing support (recall 58%), which is the harder minority-adjacent class " +
          "given the 62/38 imbalance. An ROC-AUC of 0.71 indicates the model has meaningful, though " +
          "moderate, discriminative power \u2014 well above the 0.50 random baseline."
        ),

        h2("5.3 Feature Importance"),
        new Paragraph({
          children: [img("assets_feature_importance.png", 380, 244)],
          alignment: AlignmentType.CENTER,
          spacing: { before: 100, after: 200 },
        }),
        body(
          "Technical_Score and Code_Review_Score dominate the model's decisions, together accounting for " +
          "roughly half of total feature importance. Education_Level contributes very little \u2014 the model " +
          "has learned that demonstrated performance matters far more than formal credentials, which is a " +
          "reassuring and fair outcome for an internship-evaluation tool."
        ),

        new Paragraph({ children: [new PageBreak()] }),

        h1("6. Deployed Interface"),
        body(
          "The trained model is served through a Streamlit web application (app.py) with four tabs, " +
          "built to support a program coordinator's actual workflow rather than a single input form."
        ),
        h2("6.1 Predict"),
        body(
          "A coordinator enters an intern's six numeric metrics via sliders and selects an education " +
          "level from a dropdown. The app encodes and scales the input exactly as done at training " +
          "time, runs it through the saved model, and displays a clear verdict, the predicted " +
          "probability with a progress bar, a plain-language \u201cWhy this prediction?\u201d explanation " +
          "highlighting which top features pushed the result up or down relative to typical values, " +
          "and an expandable view of the exact feature row sent to the model."
        ),
        h2("6.2 Batch Prediction"),
        body(
          "A coordinator can upload a CSV of an entire cohort and receive predictions and probabilities " +
          "for every intern at once, with a downloadable results file and a summary count of how many " +
          "interns fall into each outcome class \u2014 turning a one-at-a-time tool into something usable " +
          "for real program-level screening."
        ),
        h2("6.3 Model Comparison"),
        body(
          "Logistic Regression and a Decision Tree were trained on the identical data split as the " +
          "deployed Random Forest, and all three are shown side by side in the app (accuracy, precision, " +
          "recall, F1, ROC-AUC). This makes the model-selection decision from Section 4.1 verifiable " +
          "inside the product itself, not just asserted in this report."
        ),
        h2("6.4 Feature Importance"),
        body(
          "An interactive chart of the deployed model's feature weights, with the same plain-language " +
          "takeaway as Section 5.3, so a non-technical stakeholder can see what drives the tool's " +
          "recommendations without reading this document."
        ),
        body("Deployment link / demo: [ADD_YOUR_DEPLOYED_LINK_OR_VIDEO_HERE]", { italics: true, color: "808080" }),

        h1("7. Limitations & Future Work"),
        bullet("Dataset size (500 records) is modest; a larger, real-world sample would likely improve generalization and reduce variance in the minority-class recall."),
        bullet("69% accuracy means roughly 3 in 10 predictions are wrong \u2014 the tool should support, not replace, a coordinator's judgment."),
        bullet("Recall on the \u201cNeeds Support\u201d class (58%) is the weakest metric; techniques like class-weighting or SMOTE oversampling could be explored to catch more at-risk interns."),
        bullet("The \u201cWhy this prediction?\u201d explanation compares an intern's values to typical dataset ranges; a future version could use SHAP values for a mathematically precise, per-prediction breakdown of each feature's exact contribution."),
        bullet("Batch prediction currently expects a clean CSV with exact column names; a future version could add fuzzy column matching and inline validation feedback for messier real-world exports."),

        h1("8. Conclusion"),
        body(
          "This project demonstrates a complete, working machine learning pipeline for Zynxis \u2014 from raw " +
          "intern performance data through a trained, evaluated Random Forest model to a deployed, " +
          "user-friendly prediction tool. The model surfaces technical performance and code quality as the " +
          "dominant predictors of intern success, giving Zynxis a concrete, data-backed signal to " +
          "complement its existing internship evaluation process."
        ),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("Zynxis_Capstone_Report.docx", buf);
  console.log("Report written.");
});
