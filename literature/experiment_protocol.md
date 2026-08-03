# Experiment protocol

1. Collect public USD-M Futures data without account or trading permissions.
2. Keep raw files immutable and store processing configuration with each run.
3. Split observations chronologically: old data for training, next period for threshold selection, newest period for final test.
4. Never compute rolling baselines with future observations.
5. Keep all rows from a single pump episode in the same split.
6. Report PR-AUC, precision, recall, F1, Brier score and false alerts per day.
7. Add symbol holdout and market-regime holdout before claiming generalization.
8. Compare every complex model with adaptive rules and logistic regression.
9. Treat synthetic results only as software verification.
10. Do not connect the research output to automatic order execution.
