# Independent language-model audit: summary

- Sample: 60 Qwen-generated evaluation pairs (21 Orthopedic test, 21 Psychiatry, 18 Pulmonology; 20 per language: English, French, Spanish).
- Auditors: ChatGPT (GPT-6) and Gemini 3 Flash, web interfaces, 24 October 2026. Neither model took part in generation or validation.
- Questions: Q1 plausible, Q2 one change, Q3 original fits, Q4 edited fits (see `audit_prompt.md`).

| Auditor | Accepted (of 60) | Rate | Exact 95% interval |
|---|---|---|---|
| ChatGPT | 22 | .37 | [.25, .50] |
| Gemini | 24 | .40 | [.28, .54] |
| Both | 15 | .25 | [.15, .38] |

Agreement between the two auditors: 44 of 60 decisions, kappa = .44.

Files: `annotation_A_blank.xlsx` (the sheet given to both models), `audit_responses_chatgpt.csv`, `audit_responses_gemini.csv`. The item-to-pair mapping key is not released.
