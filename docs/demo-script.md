# Demo Script: Churn Decision Intelligence (3–5 minutes)

**Story:** score a customer → why they're flagged → what to do → what it's worth at our scale → hand the exec a one-page summary.

**Setup (before the call):** open https://churn.ameyparmarthi.com/ about a minute early so the Cloud Run API wakes from scale-to-zero. The status pill should be green; if it's amber, the app is in Demo Mode and every estimate is badged. Have a second tab on `#/simulator`.

---

## 1. Frame the problem (30 s): `#/overview`

> "A subscription business loses about 1.2% of its base a month. Retention can't call everyone, so the question isn't 'who churns?' but 'who do we contact this cycle, and is it worth it?' This is the tool an operator and an exec would use for that."

Point at the champion (LightGBM, ROC AUC 0.966, PR AUC 0.539 on a time-based Feb-2017 holdout, about 43× the base rate) and say every dollar figure is simulated on that validation window, not realised revenue.

## 2. Score a customer (45 s): `#/score` → Single

Pick a high-risk profile, such as "One-off purchase, expired, inactive", and score it. Check the band in a rehearsal first: the live model's score can differ from the profile's rule-of-thumb estimate.

> "This comes from the live model on Cloud Run. It lands in the **Contact now** band, so the suggested action is personal outreach with a retention incentive."

## 3. Why they're flagged (45 s): the "Why this score" panel

> "Operators won't act on a number they can't explain. These are the top drivers: no auto-renew, membership lapsed, few transactions. Red raises risk, green lowers it."

Then say the caveat out loud:

> "These are indicative drivers from a transparent rule of thumb, not the model's own attributions. The panel says so, and it shows the rule of thumb's estimate next to the model's score, so you can see when they disagree."

## 4. What to do at volume (30 s): Batch

Click **Use 5 samples** or paste the 60-customer JSON. Show the rank, the band and the "Top reasons" chips, then **Export CSV**.

> "The export is a contact list: rank, band, suggested action and top reasons per customer, ready to load into an outreach tool."

## 5. What it's worth at our scale (60 s): `#/simulator`

Set subscribers to the prospect's base (e.g. 1,000,000). Show K = 1,478 at validation scale (the ROI-optimal threshold of 0.68), then move K, the value per save and the save rate.

> "At $120 per save, $5 per contact and a 20% save rate, the ROI-optimal policy contacts the top 0.76% and nets about $17.7K a cycle at validation scale, a 3.4× return on contact spend. The tornado shows save rate and value per save move net value equally. We can look up the value of a subscriber, but the save rate is pure assumption, so that's what we test first."

## 6. Hand the exec a one-pager (45 s): **Executive summary**

Click **Executive summary**.

> "This is what goes to the VP: the recommendation, the key numbers, a contact plan by risk band at their subscriber count, assumptions, risks and caveats. Print to PDF or download as Markdown. The link at the bottom reopens this exact scenario."

Close on the risk line:

> "The 20% save rate is an assumption. The next step is a holdout experiment to measure it before we scale."

---

## Prepared Q&A

**"Are these real model explanations?"**

> "No, and the UI says so. The drivers come from a transparent rule of thumb over the same inputs, with each factor's impact attributed so they add up to the gap from a 4.7% baseline. That gives operators a readable reason today without claiming it's the model. The change I'd ship next is an optional `explanations` field on `/predict` and `/predict_batch`. LightGBM computes exact TreeSHAP contributions natively with `pred_contrib=True`, so it needs no new dependency, and I'd make it opt-in so big batch files don't pay the latency cost. Then the panel, the table and the CSV show the model's real attributions, and the rule of thumb stays only as the offline fallback."

**"Why not leave-one-out for the drivers?"**

> "It saturates. For a customer at 99.7% risk, removing a strong factor moves the score about 1 point, so every driver looks negligible and they don't add up. Scaling each factor's log-odds by the sigmoid's average slope fixes both problems."

**"Is the $17.7K real?"**

> "It's a simulation on the Feb-2017 validation window under stated assumptions, not realised revenue. The holdout experiment is what turns it into a measured number."

**"How do the bands map to the policy?"**

> "Contact now is score ≥ 0.68, the ROI-optimal threshold (top 1,478 at validation scale). Capacity tier runs down to 0.2058, the score at the retention team's 10,000-contact capacity. Everyone else is monitored and re-scored next cycle."
