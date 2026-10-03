# R007 frozen E005B over R006B candidates

This isolated evaluation changes only the blocker to R006B (20k soft posting
budget, M=800, 20 CPU threads). It reuses the frozen E001 LightGBM model,
historical E005 input construction, epoch-2 XLM-R checkpoint, shortlist rule,
0.60/0.40 fusion, threshold 0.51, owner margin 0.48, and official grouped
macro F0.5. No training or calibration is performed.
