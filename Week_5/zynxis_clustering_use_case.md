# Week 5: Practical Use Cases for Clustering at Zynxis

**Author:**  Syeda Mahnoor Fatima 
**Topic:** Applying K-Means and DBSCAN Clustering to Intern Performance Data  

---

## 1. What the Clusters Showed
Using the intern dataset (`zynxis_intern_performance.csv`, 500 rows) without looking at the target outcome during training, the clustering models separated the interns into three main groups that align closely with educational background and baseline experience:

- **Cluster 0 (Undergrad Cohort — 255 interns):**
  - Average Technical Score: ~72.9 / 100
  - Project Completion Rate: ~82.2%
  - High Performer / Placement Rate: **64.3%**
  - *Summary:* This is the largest group. They have solid technical skills and steady project delivery.

- **Cluster 1 (Graduate Cohort — 131 interns):**
  - Average Technical Score: ~72.1 / 100
  - Project Completion Rate: ~84.0%
  - High Performer / Placement Rate: **64.9%**
  - *Summary:* Slightly higher average project completion and consistent performance across weekly evaluations.

- **Cluster 2 (High School / Entry Cohort — 114 interns):**
  - Average Technical Score: ~69.6 / 100
  - Project Completion Rate: ~81.4%
  - High Performer / Placement Rate: **53.5%**
  - *Summary:* Lower high-performer rate compared to the other two cohorts (~11% lower), though attendance and effort metrics remain comparable.

---

## 2. What DBSCAN Outliers Revealed
DBSCAN (using $\varepsilon = 1.8, \text{min\_samples} = 5$) identified the dense areas of each cohort and flagged **72 interns (14.4%) as outliers/noise (label -1)**.

When inspecting these individual rows:
- Some interns had high prior experience (8+ months) but below average project completion rates.
- Others had lower technical test scores (under 55) but near 100% attendance and high soft skills ratings.
- These interns do not fit the standard cohort profile and would be missed by simple averages.

---

## 3. Practical Ways Zynxis Can Use These Insights

### A. Targeted Support for the Entry / High School Group (Cluster 2)
Because Cluster 2 has a 53.5% high-performer rate compared to ~64-65% for the college cohorts, Zynxis shouldn't treat all incoming interns identically. 
- Setting up a targeted 1-week technical onboarding workshop (focused on Git, debugging, and core programming fundamentals) for this group early in the program could bridge the technical gap before weekly projects start.

### B. Proactive Check-ins for DBSCAN Outliers
Instead of waiting until final evaluations to see who struggled:
- The program leads can run DBSCAN in Week 2 or 3 to flag interns with unusual patterns (e.g. high attendance but low project completion).
- Mentors can schedule a 15-minute 1-on-1 check-in with flagged interns to diagnose whether the blocker is technical confusion, time management, or tooling issues.

### C. Balanced Team Project Formation
When assigning interns to group projects or sprint teams:
- Zynxis can mix interns across clusters (pairing an intern from Cluster 0 or 1 with an intern from Cluster 2) rather than letting teams form randomly.
- This encourages peer mentoring, keeps teams balanced in technical and completion capability, and improves overall submission quality.

---

## 4. Summary
Unsupervised clustering gave us a clear picture of how intern backgrounds affect project outcomes without manual labeling. Combining K-Means for cohort-level planning and DBSCAN for flagging atypical cases gives Zynxis simple, actionable ways to improve intern support and program success rates.
