Part A: Problem Definition
Project title
Education Insights Lab: Predicting STEM Certification from Institutional Characteristics
Background
An education-focused organization wants to understand how STEM certification is distributed across institutions in the United States. It holds a dataset of institutions with their school level, ownership, Title I status, staffing, funding, student demographics, socioeconomic indicators, and broadband access. The dataset also records whether each institution is STEM-certified (stem_certified_flag).
The problem
The organization has many recorded characteristics but does not know how informative they are for identifying STEM-certified institutions. It also does not know which characteristics matter most. This project tests that using a classification model.
The analysis does not decide which schools deserve certification. Any patterns found are signals for further investigation, not evidence of what causes certification.
Why it matters
It shows whether the organization's existing data can help distinguish certified from non-certified institutions.
It identifies which characteristics (for example resources, staffing, or technology access) are most associated with certification, so the organization knows where to look next.
It shows where the model's predictions are unreliable, which tells the organization where better or additional data is needed.
Stakeholder
An education-focused organization that wants to compare institutional patterns and guide further analysis or planning.
Primary research question
How well can institutional characteristics (school level, ownership, Title I status, staffing, funding, socioeconomic context, and broadband access) predict whether an institution is STEM-certified?
Supporting questions
How common is STEM certification in the dataset, and how does it vary by school level and ownership?
Is STEM certification statistically associated with Title I status? (chi-square test of independence)
Which characteristics contribute most to the model's predictions?
How reliably does the model identify certified and non-certified institutions, and what kinds of errors does it make?
Does model performance differ across school levels or ownership types?
Approach (summary)
Task: binary classification with stem_certified_flag as the target; each row is one institution.
Models: Logistic Regression as an interpretable baseline, compared with Random Forest.
Preparation: preprocessing in a scikit-learn pipeline so that cleaning, encoding, and scaling are reproducible and applied only to training data within each split.
Evaluation: accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC. If certified institutions are a small share of the data, accuracy alone can be misleading, so precision, recall, and F1 will guide model choice.
Interpretation: model coefficients and permutation importance, reported as predictive importance, not causal effect.
Assumptions
Each row represents one institution, and stem_certified_flag is accurate and consistently recorded.
The recorded characteristics are available before or independently of certification status. Variables that may be consequences of certification will be reviewed before modeling to avoid misleading results.
Missing values (for example, SAT/ACT scores, AP/IB participation, and postsecondary enrollment) may reflect school level or data availability rather than random gaps. I will investigate them before deciding to remove, impute, or retain them.
Limitations
The project brief does not define how STEM certification is awarded or how the data were collected. I will verify this if possible.
The data are observational and institution-level, so results show associations and predictive patterns only. They cannot establish cause and effect or describe individual students.
Class imbalance or missing data may affect model performance.
Results apply to this dataset and may not generalize to other institutions or years.
