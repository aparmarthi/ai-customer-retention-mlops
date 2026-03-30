const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Amey Parmarthi";
pres.title = "AI Customer Retention & Decision Intelligence Platform";

// ── Color Palette ──────────────────────────────────────────────────────────
const C = {
  navy:      "0F1B2D",
  darkNavy:  "0A1420",
  teal:      "0D9488",
  tealLight: "14B8A6",
  mint:      "5EEAD4",
  white:     "FFFFFF",
  offWhite:  "F8FAFC",
  gray100:   "F1F5F9",
  gray300:   "CBD5E1",
  gray400:   "94A3B8",
  gray500:   "64748B",
  gray700:   "334155",
  gray900:   "0F172A",
  orange:    "F59E0B",
  red:       "EF4444",
  green:     "22C55E",
};

// ── Helpers ────────────────────────────────────────────────────────────────
const makeShadow = () => ({ type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.12 });

function addDarkSlide() {
  const s = pres.addSlide();
  s.background = { color: C.navy };
  return s;
}

function addLightSlide(title) {
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  // Top teal bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.teal } });
  // Title
  if (title) {
    s.addText(title, {
      x: 0.6, y: 0.25, w: 8.8, h: 0.55,
      fontSize: 26, fontFace: "Calibri", bold: true, color: C.gray900, margin: 0,
    });
    // Teal accent dot
    s.addShape(pres.shapes.OVAL, { x: 0.6, y: 0.95, w: 0.12, h: 0.12, fill: { color: C.teal } });
  }
  // Footer bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.325, w: 10, h: 0.3, fill: { color: C.navy } });
  s.addText("AI Customer Retention Platform  |  Amey Parmarthi", {
    x: 0.6, y: 5.325, w: 8.8, h: 0.3,
    fontSize: 8, fontFace: "Calibri", color: C.gray400, valign: "middle",
  });
  return s;
}

function card(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: opts.fill || C.white },
    shadow: makeShadow(),
    line: { color: C.gray300, width: 0.5 },
  });
  if (opts.accentTop) {
    slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h: 0.06, fill: { color: opts.accentTop } });
  }
}

function statBlock(slide, x, y, value, label, color) {
  card(slide, x, y, 2.0, 1.1, { accentTop: color || C.teal });
  slide.addText(value, {
    x, y: y + 0.15, w: 2.0, h: 0.55,
    fontSize: 28, fontFace: "Calibri", bold: true, color: color || C.teal, align: "center", margin: 0,
  });
  slide.addText(label, {
    x, y: y + 0.65, w: 2.0, h: 0.35,
    fontSize: 10, fontFace: "Calibri", color: C.gray500, align: "center", margin: 0,
  });
}

function tableRows(slide, x, y, w, headers, rows, opts = {}) {
  const colW = opts.colW || headers.map(() => w / headers.length);
  const headerRow = headers.map(h => ({
    text: h, options: { bold: true, color: C.white, fill: { color: C.navy }, fontSize: 11, fontFace: "Calibri", align: "left", valign: "middle" }
  }));
  const dataRows = rows.map((row, ri) => row.map(cell => ({
    text: String(cell),
    options: {
      fontSize: 10, fontFace: "Calibri", color: C.gray700,
      fill: { color: ri % 2 === 0 ? C.white : C.gray100 },
      align: "left", valign: "middle",
    }
  })));
  slide.addTable([headerRow, ...dataRows], {
    x, y, w, colW,
    border: { pt: 0.5, color: C.gray300 },
    rowH: opts.rowH || 0.35,
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 1: Title
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addDarkSlide();
  // Large teal accent circle (decorative)
  s.addShape(pres.shapes.OVAL, { x: 7.5, y: -1, w: 4, h: 4, fill: { color: C.teal, transparency: 85 } });
  s.addShape(pres.shapes.OVAL, { x: 8.2, y: 3.0, w: 2.5, h: 2.5, fill: { color: C.tealLight, transparency: 90 } });

  s.addText("AI CUSTOMER RETENTION", {
    x: 0.8, y: 1.0, w: 7, h: 0.7,
    fontSize: 36, fontFace: "Calibri", bold: true, color: C.white, charSpacing: 3, margin: 0,
  });
  s.addText("& Decision Intelligence Platform", {
    x: 0.8, y: 1.7, w: 7, h: 0.5,
    fontSize: 22, fontFace: "Calibri", color: C.tealLight, margin: 0,
  });
  // Teal line separator
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 2.4, w: 2.5, h: 0.04, fill: { color: C.teal } });

  s.addText("From 31 GB of Raw Data to a Production Decision Engine", {
    x: 0.8, y: 2.7, w: 7, h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: C.gray400, margin: 0,
  });

  s.addText("Amey Parmarthi  |  ML Engineering Bootcamp Capstone", {
    x: 0.8, y: 3.5, w: 7, h: 0.35,
    fontSize: 12, fontFace: "Calibri", color: C.gray400, margin: 0,
  });

  s.addText([
    { text: "GitHub: ", options: { color: C.gray500, fontSize: 10 } },
    { text: "github.com/aparmarthi/ai-customer-retention-mlops", options: { color: C.tealLight, fontSize: 10, breakLine: true } },
    { text: "Live Dashboard: ", options: { color: C.gray500, fontSize: 10 } },
    { text: "amey-churn-predictor.streamlit.app", options: { color: C.tealLight, fontSize: 10 } },
  ], { x: 0.8, y: 4.2, w: 7, h: 0.8, fontFace: "Calibri", margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 2: The Business Problem
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("The Business Problem");

  s.addText("Every Subscription Business Faces Three Questions", {
    x: 0.6, y: 1.15, w: 8.8, h: 0.35,
    fontSize: 15, fontFace: "Calibri", italic: true, color: C.gray500, margin: 0,
  });

  const questions = [
    { q: "Who will churn?", a: "Ranked probability score for every subscriber", icon: "?" },
    { q: "Who should we target?", a: "Hybrid policy: budget-bounded top-K or ROI-optimal threshold", icon: "!" },
    { q: "What's the financial impact?", a: "Simulated net ROI under configurable cost assumptions", icon: "$" },
  ];

  questions.forEach((item, i) => {
    const cy = 1.7 + i * 1.1;
    card(s, 0.6, cy, 8.8, 0.95);
    // Teal icon circle
    s.addShape(pres.shapes.OVAL, { x: 0.85, y: cy + 0.2, w: 0.55, h: 0.55, fill: { color: C.teal } });
    s.addText(item.icon, {
      x: 0.85, y: cy + 0.2, w: 0.55, h: 0.55,
      fontSize: 20, fontFace: "Calibri", bold: true, color: C.white, align: "center", valign: "middle", margin: 0,
    });
    s.addText(item.q, {
      x: 1.6, y: cy + 0.12, w: 7.5, h: 0.35,
      fontSize: 15, fontFace: "Calibri", bold: true, color: C.gray900, margin: 0,
    });
    s.addText(item.a, {
      x: 1.6, y: cy + 0.48, w: 7.5, h: 0.35,
      fontSize: 11, fontFace: "Calibri", color: C.gray500, margin: 0,
    });
  });

  s.addText("Model Probability  -->  Decision Policy  -->  Financial Outcome", {
    x: 0.6, y: 5.0, w: 8.8, h: 0.25,
    fontSize: 11, fontFace: "Consolas", bold: true, color: C.teal, align: "center", margin: 0,
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 3: The Data
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("The Data: 31 GB of Real KKBox Subscriptions");

  // Stats row
  statBlock(s, 0.6,  1.25, "31 GB",   "Raw Data Size",   C.teal);
  statBlock(s, 2.85, 1.25, "~1M",     "Subscribers",     C.tealLight);
  statBlock(s, 5.1,  1.25, "1.24%",   "Churn Rate",      C.orange);
  statBlock(s, 7.35, 1.25, "40",      "Features Built",  C.gray700);

  tableRows(s, 0.6, 2.7, 8.8,
    ["Property", "Value"],
    [
      ["Raw data size", "~31 GB (4 tables)"],
      ["Users", "~1 million subscribers"],
      ["Time range", "Jan 2015 -- Feb 2017"],
      ["Tables", "Members, Transactions, User Logs, Labels"],
      ["Processed model table", "193,205 rows, ~40 features"],
      ["Validation churn rate", "1.24% (severe class imbalance)"],
    ],
    { colW: [3.5, 5.3] }
  );

  s.addText("The full 31 GB was processed end-to-end -- not sampled, not approximated.", {
    x: 0.6, y: 5.0, w: 8.8, h: 0.25,
    fontSize: 11, fontFace: "Calibri", bold: true, italic: true, color: C.teal, align: "center", margin: 0,
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 4: Data Pipeline Architecture
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("8-Step Reproducible ETL Pipeline");

  const steps = [
    ["01", "CSV to Parquet (3.4x compression)", "DuckDB streaming"],
    ["02", "Build user-month spine", "DuckDB SQL"],
    ["03", "Aggregate transactions", "DuckDB 2-stage"],
    ["04", "Aggregate user logs", "DuckDB 2-stage"],
    ["05", "Join all to model_table (118 MB)", "pandas"],
    ["06", "Create sample data for CI", "pandas"],
    ["07", "Derive recency/tenure/frequency", "DuckDB"],
    ["08", "Create SageMaker subset", "pandas"],
  ];

  steps.forEach((step, i) => {
    const row = Math.floor(i / 2);
    const col = i % 2;
    const cx = 0.6 + col * 4.6;
    const cy = 1.25 + row * 0.95;

    card(s, cx, cy, 4.3, 0.8);
    // Step number circle
    s.addShape(pres.shapes.OVAL, { x: cx + 0.15, y: cy + 0.18, w: 0.44, h: 0.44, fill: { color: C.teal } });
    s.addText(step[0], {
      x: cx + 0.15, y: cy + 0.18, w: 0.44, h: 0.44,
      fontSize: 14, fontFace: "Calibri", bold: true, color: C.white, align: "center", valign: "middle", margin: 0,
    });
    s.addText(step[1], {
      x: cx + 0.75, y: cy + 0.1, w: 3.3, h: 0.35,
      fontSize: 11, fontFace: "Calibri", bold: true, color: C.gray900, margin: 0,
    });
    s.addText(step[2], {
      x: cx + 0.75, y: cy + 0.42, w: 3.3, h: 0.3,
      fontSize: 10, fontFace: "Calibri", color: C.gray500, margin: 0,
    });
  });

  card(s, 0.6, 5.0, 8.8, 0.25, { accentTop: C.orange });
  s.addText("Why DuckDB? pandas would OOM on 29 GB user_logs.csv. DuckDB streams it with 4 GB RAM.", {
    x: 0.8, y: 5.05, w: 8.4, h: 0.2,
    fontSize: 10, fontFace: "Calibri", color: C.gray700, margin: 0,
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 5: Train/Test Split
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("Time-Based Holdout (Not Random Split)");

  // Two comparison cards
  card(s, 0.6, 1.25, 4.3, 2.5);
  s.addText("Random Split", {
    x: 0.6, y: 1.35, w: 4.3, h: 0.35,
    fontSize: 16, fontFace: "Calibri", bold: true, color: C.red, align: "center", margin: 0,
  });
  s.addText([
    { text: "ROC-AUC: 0.9875", options: { fontSize: 13, breakLine: true } },
    { text: "PR-AUC: 0.8771", options: { fontSize: 13, breakLine: true } },
    { text: "Churn rate: ~6%", options: { fontSize: 13, breakLine: true } },
    { text: "", options: { fontSize: 8, breakLine: true } },
    { text: "Leaks temporal patterns", options: { fontSize: 11, color: C.red, italic: true } },
  ], { x: 1.0, y: 1.8, w: 3.5, h: 1.8, fontFace: "Calibri", color: C.gray700, margin: 0 });

  card(s, 5.1, 1.25, 4.3, 2.5, { accentTop: C.teal });
  s.addText("Time-Based Holdout", {
    x: 5.1, y: 1.35, w: 4.3, h: 0.35,
    fontSize: 16, fontFace: "Calibri", bold: true, color: C.teal, align: "center", margin: 0,
  });
  s.addText([
    { text: "ROC-AUC: 0.9660", options: { fontSize: 13, breakLine: true } },
    { text: "PR-AUC: 0.5392", options: { fontSize: 13, breakLine: true } },
    { text: "Churn rate: 1.24%", options: { fontSize: 13, breakLine: true } },
    { text: "", options: { fontSize: 8, breakLine: true } },
    { text: "Matches production conditions", options: { fontSize: 11, color: C.teal, italic: true } },
  ], { x: 5.5, y: 1.8, w: 3.5, h: 1.8, fontFace: "Calibri", color: C.gray700, margin: 0 });

  card(s, 0.6, 4.0, 8.8, 1.0);
  s.addText([
    { text: "Why time-based? ", options: { bold: true, color: C.gray900, fontSize: 12 } },
    { text: "Train on everything up to Jan 31, 2017. Validate on Feb 2017 (the future the model has never seen). This is how it works in production: you train on history and predict tomorrow. The lower time-based numbers are more honest.", options: { color: C.gray500, fontSize: 11 } },
  ], { x: 0.85, y: 4.1, w: 8.3, h: 0.8, fontFace: "Calibri", margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 6: Model Benchmarking
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("12 Architectures Benchmarked");

  tableRows(s, 0.6, 1.25, 8.8,
    ["#", "Model", "PR-AUC", "ROC-AUC", "Train Time"],
    [
      ["1", "LightGBM (Champion)", "0.8887", "0.9894", "~3 min"],
      ["2", "XGBoost", "0.8771", "0.9875", "~2 min"],
      ["3", "CatBoost", "0.8737", "0.9865", "~3.7 min"],
      ["4", "FT-Transformer", "0.8214", "0.9824", "~23 min"],
      ["5", "Random Forest", "0.7935", "0.9782", "~5 min"],
      ["6", "NODE", "0.7719", "0.9737", "~11 min"],
      ["7", "TabNet", "0.5233", "0.9085", "~32 min"],
    ],
    { colW: [0.5, 3.5, 1.6, 1.6, 1.6], rowH: 0.38 }
  );

  s.addText("Also evaluated: Logistic Regression, Decision Tree, FLAML AutoML, Ensemble (12 total)", {
    x: 0.6, y: 4.2, w: 8.8, h: 0.3,
    fontSize: 10, fontFace: "Calibri", italic: true, color: C.gray500, margin: 0,
  });

  card(s, 0.6, 4.55, 8.8, 0.7, { accentTop: C.orange });
  s.addText([
    { text: "Why PR-AUC as primary metric? ", options: { bold: true, color: C.gray900 } },
    { text: "With 1.24% churn, ROC-AUC looks great (0.96+) even when the model wastes outreach budget. PR-AUC directly measures how well churners are concentrated at the top of the ranked list.", options: { color: C.gray500 } },
  ], { x: 0.85, y: 4.65, w: 8.3, h: 0.5, fontSize: 11, fontFace: "Calibri", margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 7: Champion Model
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("Champion: LightGBM + FLAML AutoML");

  // Stats row — 4 cards, evenly spaced
  statBlock(s, 0.4,  1.25, "0.9660", "ROC-AUC",        C.teal);
  statBlock(s, 2.65, 1.25, "0.5392", "PR-AUC",         C.teal);
  statBlock(s, 4.9,  1.25, "43.5x",  "Lift vs Base",   C.orange);
  statBlock(s, 7.15, 1.25, "$17.7K", "Net ROI",        C.green);

  s.addText("FLAML independently confirmed LightGBM as the best architecture", {
    x: 0.6, y: 2.55, w: 8.8, h: 0.3,
    fontSize: 12, fontFace: "Calibri", italic: true, color: C.gray500, margin: 0,
  });

  // Two-column: metrics table + FLAML discoveries
  card(s, 0.6, 2.95, 4.3, 2.1);
  s.addText("Key Metrics", {
    x: 0.8, y: 3.0, w: 3.9, h: 0.3,
    fontSize: 13, fontFace: "Calibri", bold: true, color: C.gray900, margin: 0,
  });
  s.addText([
    { text: "Precision @ top-10K:  18.0%  (3x vs churn rate)", options: { fontSize: 10, breakLine: true } },
    { text: "Recall @ top-10K:  75.0%", options: { fontSize: 10, breakLine: true } },
    { text: "ROI-optimal:  $17,666  (1,478 users targeted)", options: { fontSize: 10, breakLine: true } },
    { text: "Train time:  ~3 minutes", options: { fontSize: 10, breakLine: true } },
    { text: "Inference:  <10ms per record", options: { fontSize: 10 } },
  ], { x: 0.8, y: 3.35, w: 3.9, h: 1.5, fontFace: "Calibri", color: C.gray700, margin: 0 });

  card(s, 5.1, 2.95, 4.3, 2.1, { accentTop: C.teal });
  s.addText("Key FLAML Discoveries", {
    x: 5.3, y: 3.0, w: 3.9, h: 0.3,
    fontSize: 13, fontFace: "Calibri", bold: true, color: C.gray900, margin: 0,
  });
  s.addText([
    { text: "num_leaves = 1,212 (vs manual 64)", options: { fontSize: 10, breakLine: true } },
    { text: "  Much deeper trees generalize better", options: { fontSize: 9, italic: true, color: C.gray500, breakLine: true } },
    { text: "reg_alpha = 0.56", options: { fontSize: 10, breakLine: true } },
    { text: "  Stronger L1 regularization", options: { fontSize: 9, italic: true, color: C.gray500, breakLine: true } },
    { text: "learning_rate = 0.036 (146 iterations)", options: { fontSize: 10 } },
  ], { x: 5.3, y: 3.35, w: 3.9, h: 1.5, fontFace: "Calibri", color: C.gray700, margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 8: SHAP Explainability
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("Top Churn Drivers -- Fully Explainable (SHAP)");

  s.addText("[Insert: reports/shap_summary.png]", {
    x: 0.6, y: 1.2, w: 4.0, h: 2.5,
    fontSize: 11, fontFace: "Calibri", color: C.gray400, align: "center", valign: "middle",
    fill: { color: C.gray100 }, line: { color: C.gray300, width: 1, dashType: "dash" },
  });

  tableRows(s, 4.9, 1.2, 4.7,
    ["Rank", "Feature", "Business Signal"],
    [
      ["1", "auto_renew_rate", "Auto-renewal opt-in"],
      ["2", "cancel_rate", "Historical cancellations"],
      ["3", "plan_list_price_max", "Price sensitivity"],
      ["4", "log_last_date", "Recency of activity"],
      ["5", "membership_expire_date_max", "Subscription horizon"],
    ],
    { colW: [0.6, 1.8, 2.3], rowH: 0.38 }
  );

  card(s, 0.6, 4.1, 8.8, 0.85);
  s.addText([
    { text: "SHAP confirms the model learned causally plausible signals -- ", options: { bold: true, color: C.gray900 } },
    { text: "not spurious correlations. Every prediction is decomposable into per-feature contributions, enabling stakeholders to understand and trust the model's decisions.", options: { color: C.gray500 } },
  ], { x: 0.85, y: 4.2, w: 8.3, h: 0.65, fontSize: 11, fontFace: "Calibri", margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 9: Decision Policy & ROI
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("Two Decision Policies -- Not Just Predictions");

  // Policy 1
  card(s, 0.6, 1.25, 4.3, 2.0, { accentTop: C.teal });
  s.addText("Policy 1: Ops-Friendly Top-K", {
    x: 0.8, y: 1.4, w: 3.9, h: 0.3,
    fontSize: 14, fontFace: "Calibri", bold: true, color: C.teal, margin: 0,
  });
  s.addText([
    { text: "Target the top 10,000 highest-risk subscribers each month", options: { fontSize: 11, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "Precision: 18%  |  Recall: 75%", options: { fontSize: 11, bold: true, breakLine: true } },
    { text: "Fixed budget, no threshold tuning needed", options: { fontSize: 10, italic: true, color: C.gray500 } },
  ], { x: 0.8, y: 1.8, w: 3.9, h: 1.3, fontFace: "Calibri", color: C.gray700, margin: 0 });

  // Policy 2
  card(s, 5.1, 1.25, 4.3, 2.0, { accentTop: C.orange });
  s.addText("Policy 2: ROI-Optimal Threshold", {
    x: 5.3, y: 1.4, w: 3.9, h: 0.3,
    fontSize: 14, fontFace: "Calibri", bold: true, color: C.orange, margin: 0,
  });
  s.addText([
    { text: "Target subscribers above probability 0.68", options: { fontSize: 11, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "Precision: 70.6%  |  Contacts: 1,478", options: { fontSize: 11, bold: true, breakLine: true } },
    { text: "Net ROI: $17,666", options: { fontSize: 12, bold: true, color: C.green } },
  ], { x: 5.3, y: 1.8, w: 3.9, h: 1.3, fontFace: "Calibri", color: C.gray700, margin: 0 });

  // ROI formula
  card(s, 0.6, 3.5, 8.8, 0.5);
  s.addText("Net ROI  =  (TP x save_rate x churn_cost) - (N_targeted x intervention_cost)", {
    x: 0.6, y: 3.55, w: 8.8, h: 0.4,
    fontSize: 13, fontFace: "Consolas", bold: true, color: C.teal, align: "center", margin: 0,
  });

  // Scenario table
  tableRows(s, 0.6, 4.2, 8.8,
    ["Scenario", "Cost/Contact", "Save Rate", "Net Result"],
    [
      ["Low-cost outreach", "$0.50", "12%", "~$12,200"],
      ["Incentive offers", "$10.00", "20%", "Cost-sensitive"],
      ["ROI-optimal threshold", "$5.00", "20%", "$17,666"],
    ],
    { colW: [3.0, 2.0, 1.8, 2.0] }
  );
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 10: MLflow Experiment Tracking
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("Full Experiment Governance (MLflow)");

  s.addText("All 12 model runs tracked in MLflow experiment: kkbox_churn", {
    x: 0.6, y: 1.2, w: 8.8, h: 0.3,
    fontSize: 12, fontFace: "Calibri", italic: true, color: C.gray500, margin: 0,
  });

  const items = [
    { title: "Parameters", desc: "Split policy, feature version, all hyperparameters (auto-flattened)" },
    { title: "Metrics", desc: "ROC-AUC, PR-AUC, F1, Precision@K, Recall@K" },
    { title: "Artifacts", desc: "model.pkl, scored validation set, threshold sweep, SHAP plots" },
  ];

  items.forEach((item, i) => {
    const cy = 1.7 + i * 0.9;
    card(s, 0.6, cy, 8.8, 0.75, { accentTop: C.teal });
    s.addText(item.title, {
      x: 0.85, y: cy + 0.1, w: 2.0, h: 0.3,
      fontSize: 14, fontFace: "Calibri", bold: true, color: C.teal, margin: 0,
    });
    s.addText(item.desc, {
      x: 0.85, y: cy + 0.4, w: 8.3, h: 0.25,
      fontSize: 11, fontFace: "Calibri", color: C.gray700, margin: 0,
    });
  });

  card(s, 0.6, 4.5, 8.8, 0.6);
  s.addText([
    { text: "Key design: ", options: { bold: true, color: C.gray900 } },
    { text: "Atomic artifact bundles (every metric traces to exact data) | Split method audit trail | Standardized naming across all 12 experiments", options: { color: C.gray500 } },
  ], { x: 0.85, y: 4.58, w: 8.3, h: 0.45, fontSize: 11, fontFace: "Calibri", margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 11: Serving Architecture
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("Production API: FastAPI + Docker");

  // Endpoint table
  tableRows(s, 0.6, 1.25, 5.5,
    ["Endpoint", "Method", "Purpose"],
    [
      ["/health", "GET", "Model status, artifact paths"],
      ["/predict", "POST", "Single-record churn + threshold"],
      ["/predict_batch", "POST", "Batch scoring (JSON/CSV) + top-K"],
    ],
    { colW: [1.5, 1.0, 3.0], rowH: 0.38 }
  );

  // Production features
  card(s, 6.4, 1.25, 3.2, 2.8, { accentTop: C.teal });
  s.addText("Production Features", {
    x: 6.55, y: 1.4, w: 2.9, h: 0.3,
    fontSize: 13, fontFace: "Calibri", bold: true, color: C.teal, margin: 0,
  });
  s.addText([
    { text: "Pydantic validation", options: { bullet: true, fontSize: 10, breakLine: true } },
    { text: "Two serving policies", options: { bullet: true, fontSize: 10, breakLine: true } },
    { text: "Feature alignment", options: { bullet: true, fontSize: 10, breakLine: true } },
    { text: "Structured logging", options: { bullet: true, fontSize: 10, breakLine: true } },
    { text: "Prediction logging", options: { bullet: true, fontSize: 10, breakLine: true } },
    { text: "16 pytest tests", options: { bullet: true, fontSize: 10 } },
  ], { x: 6.55, y: 1.8, w: 2.9, h: 2.0, fontFace: "Calibri", color: C.gray700, margin: 0 });

  // Docker command
  card(s, 0.6, 3.0, 5.5, 0.55);
  s.addText("docker compose up --build", {
    x: 0.8, y: 3.05, w: 4.0, h: 0.22,
    fontSize: 14, fontFace: "Consolas", bold: true, color: C.teal, margin: 0,
  });
  s.addText("API: localhost:8000/docs  |  Dashboard: localhost:8501", {
    x: 0.8, y: 3.3, w: 5.0, h: 0.2,
    fontSize: 10, fontFace: "Consolas", color: C.gray500, margin: 0,
  });

  // Live link
  card(s, 0.6, 3.8, 8.8, 0.5, { accentTop: C.green });
  s.addText([
    { text: "Live now: ", options: { bold: true, color: C.gray900 } },
    { text: "amey-churn-predictor.streamlit.app", options: { color: C.teal } },
  ], { x: 0.85, y: 3.9, w: 8.3, h: 0.3, fontSize: 12, fontFace: "Calibri", margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 12: Streamlit Dashboard
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("Stakeholder-Facing Dashboard (Streamlit)");

  s.addText("[Insert: Screenshot of Streamlit app]", {
    x: 0.6, y: 1.2, w: 4.5, h: 3.0,
    fontSize: 11, fontFace: "Calibri", color: C.gray400, align: "center", valign: "middle",
    fill: { color: C.gray100 }, line: { color: C.gray300, width: 1, dashType: "dash" },
  });

  const tabs = [
    { name: "Single Prediction", desc: "Enter user features, get churn probability + action label" },
    { name: "Batch Scoring", desc: "Upload CSV, score all users, download results with ranks" },
    { name: "ROI Simulator", desc: "Adjust cost assumptions with sliders, see real-time ROI impact" },
  ];

  tabs.forEach((tab, i) => {
    const cy = 1.2 + i * 1.0;
    card(s, 5.4, cy, 4.2, 0.85, { accentTop: C.teal });
    s.addText(tab.name, {
      x: 5.6, y: cy + 0.12, w: 3.8, h: 0.28,
      fontSize: 13, fontFace: "Calibri", bold: true, color: C.gray900, margin: 0,
    });
    s.addText(tab.desc, {
      x: 5.6, y: cy + 0.42, w: 3.8, h: 0.35,
      fontSize: 10, fontFace: "Calibri", color: C.gray500, margin: 0,
    });
  });

  card(s, 0.6, 4.45, 8.8, 0.5);
  s.addText("The ROI Simulator lets non-technical stakeholders explore trade-offs without touching the model", {
    x: 0.85, y: 4.52, w: 8.3, h: 0.35,
    fontSize: 11, fontFace: "Calibri", italic: true, color: C.gray500, margin: 0,
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 13: CI/CD
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("Docker + GitHub Actions CI/CD");

  // Docker card
  card(s, 0.6, 1.25, 4.3, 2.5, { accentTop: C.teal });
  s.addText("Docker Image (~400 MB)", {
    x: 0.8, y: 1.4, w: 3.9, h: 0.3,
    fontSize: 14, fontFace: "Calibri", bold: true, color: C.teal, margin: 0,
  });
  s.addText([
    { text: "python:3.11-slim base", options: { bullet: true, fontSize: 10, breakLine: true } },
    { text: "Serving-only deps (requirements-serve.txt)", options: { bullet: true, fontSize: 10, breakLine: true } },
    { text: "Non-root appuser (security)", options: { bullet: true, fontSize: 10, breakLine: true } },
    { text: "HEALTHCHECK for orchestrators", options: { bullet: true, fontSize: 10, breakLine: true } },
    { text: "Model artifacts frozen in image", options: { bullet: true, fontSize: 10 } },
  ], { x: 0.8, y: 1.8, w: 3.9, h: 1.8, fontFace: "Calibri", color: C.gray700, margin: 0 });

  // CI/CD Pipeline - vertical flow
  card(s, 5.1, 1.25, 4.3, 2.5, { accentTop: C.orange });
  s.addText("CI/CD Pipeline (GitHub Actions)", {
    x: 5.3, y: 1.4, w: 3.9, h: 0.3,
    fontSize: 14, fontFace: "Calibri", bold: true, color: C.orange, margin: 0,
  });

  const pipeline = [
    { step: "1. TEST", detail: "pytest test_api.py (16 tests)", color: C.teal },
    { step: "2. BUILD", detail: "Docker build + push to ECR", color: C.orange },
    { step: "3. DEPLOY", detail: "ECS Fargate rolling deploy", color: C.green },
  ];
  pipeline.forEach((p, i) => {
    const py = 1.85 + i * 0.55;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y: py, w: 0.12, h: 0.35, fill: { color: p.color } });
    s.addText(p.step, {
      x: 5.65, y: py, w: 1.5, h: 0.35,
      fontSize: 11, fontFace: "Consolas", bold: true, color: C.gray900, margin: 0, valign: "middle",
    });
    s.addText(p.detail, {
      x: 7.1, y: py, w: 2.1, h: 0.35,
      fontSize: 10, fontFace: "Calibri", color: C.gray500, margin: 0, valign: "middle",
    });
  });

  card(s, 0.6, 4.0, 8.8, 0.5);
  s.addText("git push main  -->  tests pass  -->  image built  -->  deployed  -->  ALB health check: /health", {
    x: 0.85, y: 4.05, w: 8.3, h: 0.4,
    fontSize: 11, fontFace: "Consolas", color: C.teal, align: "center", margin: 0,
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 14: SageMaker
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("Cloud Training Validation (AWS SageMaker)");

  // Comparison cards
  card(s, 0.6, 1.25, 4.3, 1.8);
  s.addText("Local (Champion)", {
    x: 0.8, y: 1.35, w: 3.9, h: 0.3,
    fontSize: 14, fontFace: "Calibri", bold: true, color: C.teal, margin: 0,
  });
  s.addText([
    { text: "ROC-AUC: 0.9660", options: { fontSize: 12, bold: true, breakLine: true } },
    { text: "PR-AUC: 0.5392", options: { fontSize: 12, bold: true, breakLine: true } },
    { text: "Purpose: Rapid iteration, 12 experiments", options: { fontSize: 10, color: C.gray500, breakLine: true } },
    { text: "Cost: Free (local hardware)", options: { fontSize: 10, color: C.gray500 } },
  ], { x: 0.8, y: 1.75, w: 3.9, h: 1.2, fontFace: "Calibri", color: C.gray700, margin: 0 });

  card(s, 5.1, 1.25, 4.3, 1.8, { accentTop: C.orange });
  s.addText("SageMaker (Validation)", {
    x: 5.3, y: 1.35, w: 3.9, h: 0.3,
    fontSize: 14, fontFace: "Calibri", bold: true, color: C.orange, margin: 0,
  });
  s.addText([
    { text: "ROC-AUC: 0.9484", options: { fontSize: 12, bold: true, breakLine: true } },
    { text: "PR-AUC: 0.4707", options: { fontSize: 12, bold: true, breakLine: true } },
    { text: "Purpose: Cloud workflow proof", options: { fontSize: 10, color: C.gray500, breakLine: true } },
    { text: "Cost: Pay-per-minute EC2", options: { fontSize: 10, color: C.gray500 } },
  ], { x: 5.3, y: 1.75, w: 3.9, h: 1.2, fontFace: "Calibri", color: C.gray700, margin: 0 });

  // What was demonstrated
  card(s, 0.6, 3.3, 8.8, 1.6);
  s.addText("One controlled job demonstrated:", {
    x: 0.8, y: 3.4, w: 8.4, h: 0.3,
    fontSize: 13, fontFace: "Calibri", bold: true, color: C.gray900, margin: 0,
  });
  s.addText([
    { text: "S3 data ingestion (train channel)", options: { bullet: true, fontSize: 11, breakLine: true } },
    { text: "Script Mode execution (train.py)", options: { bullet: true, fontSize: 11, breakLine: true } },
    { text: "Artifact packaging (model.tar.gz --> S3)", options: { bullet: true, fontSize: 11, breakLine: true } },
    { text: "Model Registry integration (versioned, approval-gated)", options: { bullet: true, fontSize: 11 } },
  ], { x: 0.8, y: 3.75, w: 8.4, h: 1.1, fontFace: "Calibri", color: C.gray700, margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 15: Deployment Architecture
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("All-AWS Production Architecture");

  // Architecture flow - 4 columns
  const cols = [
    { title: "Data", items: ["KKBox DB (31 GB)", "DuckDB 8-step ETL", "S3 Bucket"], color: C.teal },
    { title: "Training", items: ["SageMaker LightGBM", "MLflow Tracking", "Model Registry"], color: C.orange },
    { title: "Serving", items: ["ECS Fargate (Docker)", "FastAPI", "ALB Load Balancer"], color: C.green },
    { title: "Monitoring", items: ["CloudWatch Metrics", "Latency / Errors / Drift", "SNS Alerts"], color: C.red },
  ];

  cols.forEach((col, i) => {
    const cx = 0.4 + i * 2.4;
    card(s, cx, 1.2, 2.15, 2.8, { accentTop: col.color });
    s.addText(col.title, {
      x: cx, y: 1.35, w: 2.15, h: 0.35,
      fontSize: 15, fontFace: "Calibri", bold: true, color: col.color, align: "center", margin: 0,
    });
    s.addText(col.items.map((item, j) => ({
      text: item,
      options: { fontSize: 10, breakLine: j < col.items.length - 1, color: C.gray700 }
    })), { x: cx + 0.15, y: 1.8, w: 1.85, h: 1.8, fontFace: "Calibri", margin: 0, valign: "top", align: "center" });

    // Arrow between columns
    if (i < 3) {
      s.addText("-->", {
        x: cx + 2.05, y: 2.3, w: 0.45, h: 0.3,
        fontSize: 16, fontFace: "Consolas", bold: true, color: C.teal, align: "center", valign: "middle", margin: 0,
      });
    }
  });

  // CI/CD bar
  card(s, 0.4, 4.25, 9.2, 0.5, { accentTop: C.teal });
  s.addText("CI/CD: GitHub Actions  -->  pytest  -->  Docker build  -->  ECR push  -->  ECS deploy", {
    x: 0.6, y: 4.35, w: 8.8, h: 0.3,
    fontSize: 11, fontFace: "Consolas", color: C.gray700, align: "center", margin: 0,
  });

  s.addText("Estimated cost: ~$26/month at low-to-moderate inference volume", {
    x: 0.6, y: 4.9, w: 8.8, h: 0.25,
    fontSize: 11, fontFace: "Calibri", bold: true, italic: true, color: C.teal, align: "center", margin: 0,
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 16: Monitoring
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("Monitoring & Model Care");

  tableRows(s, 0.6, 1.25, 8.8,
    ["Layer", "What", "How", "Alert Threshold"],
    [
      ["Infrastructure", "Latency, errors, memory", "CloudWatch + ALB", "p95 > 500ms"],
      ["Model", "Prediction drift", "KS-test monthly", "Mean shift > 2 std"],
      ["Business", "Precision@K, ROI", "Actuals join (30-day lag)", "ROC-AUC < 0.90"],
    ],
    { colW: [1.5, 2.2, 2.5, 2.6] }
  );

  s.addText("Retraining Triggers", {
    x: 0.6, y: 2.8, w: 4.0, h: 0.3,
    fontSize: 14, fontFace: "Calibri", bold: true, color: C.gray900, margin: 0,
  });

  tableRows(s, 0.6, 3.15, 8.8,
    ["Trigger", "Condition", "Response"],
    [
      ["Scheduled", "Monthly", "Full pipeline: pull --> train --> evaluate --> deploy"],
      ["Performance", "ROC-AUC < 0.90", "Emergency retrain within 48 hours"],
      ["Drift", ">3 features drifting (KS p<0.01)", "Investigate, retrain if metrics degrade"],
    ],
    { colW: [1.5, 2.5, 4.8] }
  );

  card(s, 0.6, 4.55, 8.8, 0.5, { accentTop: C.green });
  s.addText([
    { text: "Rollback: ", options: { bold: true, color: C.gray900 } },
    { text: "Every Docker image tagged with git SHA. ECS keeps previous task definitions. Rollback time: < 5 minutes.", options: { color: C.gray500 } },
  ], { x: 0.85, y: 4.63, w: 8.3, h: 0.35, fontSize: 11, fontFace: "Calibri", margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 17: Scaling Path
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("From 1M Users to 1B");

  tableRows(s, 0.6, 1.25, 8.8,
    ["Scale", "Users", "Raw Data", "Compute", "Change Required"],
    [
      ["Current", "1M", "31 GB", "DuckDB (local)", "None"],
      ["10x", "10M", "310 GB", "DuckDB (local)", "None"],
      ["100x", "100M", "3 TB", "Spark SQL", "Swap engine"],
      ["1000x", "1B", "31 TB", "BigQuery/Spark", "Swap engine"],
    ],
    { colW: [1.0, 1.2, 1.4, 2.4, 2.8] }
  );

  card(s, 0.6, 3.3, 8.8, 1.2);
  s.addText([
    { text: "The architecture doesn't require redesign -- only swapping the execution engine.", options: { fontSize: 14, bold: true, color: C.teal, breakLine: true } },
    { text: "", options: { fontSize: 8, breakLine: true } },
    { text: "The 2-stage aggregation pattern, Parquet format, and LightGBM all scale linearly. DuckDB handles 31 GB on a laptop today. Spark handles 31 TB in the cloud tomorrow. Same SQL, same feature logic.", options: { fontSize: 11, color: C.gray500 } },
  ], { x: 0.85, y: 3.4, w: 8.3, h: 1.0, fontFace: "Calibri", margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 18: Competencies
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addLightSlide("Full-Stack ML Engineering Competencies");

  const competencies = [
    { area: "Data Engineering", evidence: "8-step pipeline, 31 GB DuckDB, Parquet storage" },
    { area: "Model Development", evidence: "12 architectures, FLAML AutoML, time-based eval" },
    { area: "MLOps", evidence: "MLflow tracking, SageMaker validation, Model Registry" },
    { area: "Evaluation", evidence: "PR-AUC, threshold sweep, ROI simulation" },
    { area: "Explainability", evidence: "SHAP analysis, business signal mapping" },
    { area: "API Engineering", evidence: "FastAPI + Pydantic, two policies, 16 tests" },
    { area: "Containerization", evidence: "Docker slim, HEALTHCHECK, docker-compose" },
    { area: "CI/CD", evidence: "GitHub Actions: test --> build --> deploy" },
    { area: "Cloud", evidence: "SageMaker + Model Registry, ECS Fargate arch" },
    { area: "Business Translation", evidence: "ROI framework, cost scenarios, A/B design" },
  ];

  competencies.forEach((c, i) => {
    const row = Math.floor(i / 2);
    const col = i % 2;
    const cx = 0.6 + col * 4.6;
    const cy = 1.15 + row * 0.72;

    card(s, cx, cy, 4.3, 0.62);
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: cy, w: 0.08, h: 0.62, fill: { color: C.teal } });
    s.addText(c.area, {
      x: cx + 0.2, y: cy + 0.05, w: 4.0, h: 0.25,
      fontSize: 11, fontFace: "Calibri", bold: true, color: C.gray900, margin: 0,
    });
    s.addText(c.evidence, {
      x: cx + 0.2, y: cy + 0.3, w: 4.0, h: 0.25,
      fontSize: 9, fontFace: "Calibri", color: C.gray500, margin: 0,
    });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 19: Live Demo
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addDarkSlide();
  s.addShape(pres.shapes.OVAL, { x: -1, y: 3, w: 3, h: 3, fill: { color: C.teal, transparency: 88 } });

  s.addText("LIVE DEMO", {
    x: 0.8, y: 0.5, w: 8.4, h: 0.6,
    fontSize: 36, fontFace: "Calibri", bold: true, color: C.white, charSpacing: 5, margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 1.2, w: 2.0, h: 0.04, fill: { color: C.teal } });

  // API Demo
  card(s, 0.8, 1.6, 4.0, 1.6, { fill: "1A2A3E", accentTop: C.teal });
  s.addText("API (Swagger UI)", {
    x: 1.0, y: 1.75, w: 3.6, h: 0.3,
    fontSize: 15, fontFace: "Calibri", bold: true, color: C.tealLight, margin: 0,
  });
  s.addText([
    { text: "1. Click POST /predict --> Try it out", options: { fontSize: 10, breakLine: true } },
    { text: "2. Paste a feature record", options: { fontSize: 10, breakLine: true } },
    { text: "3. See churn probability + action label", options: { fontSize: 10 } },
  ], { x: 1.0, y: 2.15, w: 3.6, h: 0.9, fontFace: "Calibri", color: C.gray400, margin: 0 });

  // Dashboard Demo
  card(s, 5.2, 1.6, 4.0, 1.6, { fill: "1A2A3E", accentTop: C.orange });
  s.addText("Dashboard (Streamlit)", {
    x: 5.4, y: 1.75, w: 3.6, h: 0.3,
    fontSize: 15, fontFace: "Calibri", bold: true, color: C.orange, margin: 0,
  });
  s.addText([
    { text: "Tab 1: Single prediction with JSON", options: { fontSize: 10, breakLine: true } },
    { text: "Tab 2: Upload CSV for batch scoring", options: { fontSize: 10, breakLine: true } },
    { text: "Tab 3: ROI simulator with sliders", options: { fontSize: 10 } },
  ], { x: 5.4, y: 2.15, w: 3.6, h: 0.9, fontFace: "Calibri", color: C.gray400, margin: 0 });

  // Links
  s.addText("amey-churn-predictor.streamlit.app", {
    x: 0.8, y: 3.6, w: 8.4, h: 0.35,
    fontSize: 16, fontFace: "Consolas", color: C.tealLight, align: "center", margin: 0,
  });

  s.addText("github.com/aparmarthi/ai-customer-retention-mlops", {
    x: 0.8, y: 4.1, w: 8.4, h: 0.3,
    fontSize: 12, fontFace: "Consolas", color: C.gray400, align: "center", margin: 0,
  });

  // Local run command
  card(s, 2.0, 4.6, 6.0, 0.5, { fill: "1A2A3E" });
  s.addText("docker compose up --build", {
    x: 2.0, y: 4.65, w: 6.0, h: 0.2,
    fontSize: 13, fontFace: "Consolas", bold: true, color: C.tealLight, align: "center", margin: 0,
  });
  s.addText("API: localhost:8000/docs  |  Dashboard: localhost:8501", {
    x: 2.0, y: 4.88, w: 6.0, h: 0.18,
    fontSize: 9, fontFace: "Consolas", color: C.gray500, align: "center", margin: 0,
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 20: Key Takeaways
// ════════════════════════════════════════════════════════════════════════════
{
  const s = addDarkSlide();
  s.addShape(pres.shapes.OVAL, { x: 8, y: -1.5, w: 4, h: 4, fill: { color: C.teal, transparency: 88 } });

  s.addText("KEY TAKEAWAYS", {
    x: 0.8, y: 0.3, w: 8.4, h: 0.55,
    fontSize: 30, fontFace: "Calibri", bold: true, color: C.white, charSpacing: 4, margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 0.9, w: 2.0, h: 0.04, fill: { color: C.teal } });

  const takeaways = [
    { num: "01", title: "Deployment is harder than training", body: "The model was done in Step 4. Steps 5-12 are everything else." },
    { num: "02", title: "The right metric changes everything", body: "Switching to PR-AUC under 1.24% imbalance changed which models looked good." },
    { num: "03", title: "ML models don't make decisions -- policies do", body: "A churn probability is useless without a targeting policy." },
    { num: "04", title: "Scale decisions should be lazy", body: "DuckDB handles 31 GB on a laptop. Choose the simplest tool that works today." },
    { num: "05", title: "Build the infrastructure yourself", body: "FastAPI + Docker + CI/CD from scratch forces understanding of every layer." },
  ];

  takeaways.forEach((t, i) => {
    const ty = 1.1 + i * 0.78;
    s.addText(t.num, {
      x: 0.8, y: ty, w: 0.5, h: 0.3,
      fontSize: 16, fontFace: "Calibri", bold: true, color: C.teal, margin: 0,
    });
    s.addText(t.title, {
      x: 1.4, y: ty, w: 8.0, h: 0.3,
      fontSize: 13, fontFace: "Calibri", bold: true, color: C.white, margin: 0,
    });
    s.addText(t.body, {
      x: 1.4, y: ty + 0.3, w: 8.0, h: 0.3,
      fontSize: 9, fontFace: "Calibri", color: C.gray400, margin: 0,
    });
  });

  // Thank you — positioned after takeaways with clear gap
  s.addShape(pres.shapes.RECTANGLE, { x: 3.5, y: 5.05, w: 3.0, h: 0.03, fill: { color: C.teal } });
  s.addText("Thank You", {
    x: 0.8, y: 5.15, w: 8.4, h: 0.35,
    fontSize: 16, fontFace: "Calibri", color: C.tealLight, align: "center", margin: 0,
  });
}

// ── Generate ───────────────────────────────────────────────────────────────
pres.writeFile({ fileName: "reports/step_12_capstone_presentation.pptx" })
  .then(() => console.log("Created: reports/step_12_capstone_presentation.pptx"))
  .catch(err => console.error("Error:", err));
