# Audit prompt

This is the exact prompt given to each language model in a fresh chat (no memory, no earlier conversation), with only `annotation_A_blank.xlsx` attached. The same prompt was used for both auditors.

Auditors: ChatGPT (GPT-6) and Gemini 3 Flash, both through their web interfaces, on 24 October 2026.

```
You are an independent auditor for a research study on clinical search. Please judge each item in the attached workbook honestly and on its own merits. Do not guess how the data were produced, do not try to be agreeable, and do not assume an item is good or bad before reading it. If something is wrong, say so.

The workbook has three sheets. "Instructions" explains the study and the four questions (in English and Chinese). "Examples" shows worked examples. "Items" holds the 60 items to judge. Read Instructions and Examples first and follow them exactly.

For every row on the "Items" sheet, fill these columns:
- Q1 plausible: is the edited passage medically plausible?
- Q2 one change: does the edited passage differ from the original only by the marked change, and does the Attribute label describe that change?
- Q3 original fits: does the original passage answer the original query better than the edited passage does? Answer 0 if they answer it equally well, or if the edited passage answers it better.
- Q4 edited fits: does the edited passage answer the edited query better than the original passage does, and do the two queries ask about the same topic? The edited query is NOT supposed to state the new value, so do not mark it down for that. Judge only whether the fact it asks for appears in the edited passage and not in the original. Answer 0 if both passages answer it equally well.
- Note: one short sentence giving the reason, especially for every 0 or ?.

Rules:
1. Use only 1 (yes), 0 (no) or ? (cannot judge). You can read English, French and Spanish, so use ? only if the excerpt truly lacks the information needed.
2. Judge from the excerpts and the two queries only. Do not use outside knowledge to decide what the queries "should" ask.
3. Judge each row independently. Do not look for patterns across rows, and do not try to balance your answers.
4. Work through the rows in batches of 20 and re-read each question before each batch.
5. Do not change any other column or sheet, and do not add or remove rows.
6. When finished, check that all 60 rows have all four answers and a Note. Report any row you could not complete.
7. Save the completed workbook with the same sheet names and columns, named annotation_A_completed_[your model name].xlsx, and give me a download link. If you cannot make an Excel file, give me a CSV with the same columns.
8. At the end, tell me in two sentences which instruction you found unclear, if any. Do not summarize the results.

Do not tell me what you think the study is trying to show.
```

## Acceptance rule

A pair is acceptable only if Q1 and Q2 are both 1 and every answered Q3 and Q4 is 1. The audit is a language-model audit, not a human judgment.
