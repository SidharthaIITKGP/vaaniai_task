# Requirements verified against The_Relay_Assignment_Brief.pdf

The brief is four pages. The deliverable is at most two pages of writing, plus
code, raw request results and the real assistant conversation history. The time
budget is four days from receipt; receipt time is not established by the file.

## Required writing order (page 2)

1. What I built and why
2. The real service and my stand-in
3. Results
4. Where the time goes
5. When a limit is hit
6. Retrieval: quality versus cost
7. How this library will mislead a naive retriever
8. Retries
9. What the person sees
10. What I'm unsure about, what I left out, what I learned
11. Your feedback on Vaani
12. How I worked with the assistant

The last section needs the three influential prompts, an assistant mistake caught,
an overruled decision, and the user's own contribution. The assistant can recover
the visible transcript and identify actual mistakes, but cannot invent the user's
personal views or claim that assistant-written first-person text is their account.

## Results contract (pages 3–4)

`python -m relay.assignment_export` produces a new `submission_pdf_aligned/`
directory. `results.csv` is **one row per request**, not the previous summary table.
The leading columns are exactly:

```text
request_id,scenario,source,start_time,first_token_ms,total_ms,status,retries,tokens_in,tokens_out,retrieval_setting,answer_grade
```

Source is `real` or `stand-in`. Status is `ok`, `rate_limited`, `timeout`, or
`error`. Grades, when reviewed, are `correct`, `wrong`, `refused_correctly`, or
`refused_wrongly`. Ungraded rows are blank and explicitly marked `not_measured`
in the extra `grade_status` column. No answer grade is invented for a timing-only
stand-in. Extra diagnostics preserve queue length, cost estimate, original status,
data file provenance and virtual/measured time, making table calculations auditable.

Required tables are exported separately as `table1.csv`, `table2a.csv`,
`table2b.csv`, and `table3.csv`. Durations are milliseconds except scenario
duration in seconds. Percentages have one decimal. Unknown values say
`not measured`. Final write-up must include these tables, not substitute prose.

## Outstanding requirements

- Complete and independently grade 15+ matched questions before architecture selection.
- Evaluate final answer quality, tokens and TTFT for at least three retrieval settings.
- Run the actual pipeline under load through a stand-in with configurable slow calls,
  failures, capacity refusals, streaming and increasing service delay under load.
  Current virtual service/bundle simulations are diagnostic; they do not complete this requirement.
- Verify tokens/day, actual in-flight limit and actual recovery, or keep `not measured`.
- Produce final two-page report with all headings and tables after quality work completes.
- Obtain the user's own feedback and contribution/reflection; do not invent it.

## Website review observations (assistant, not the user's testimony)

Reviewed https://www.vaani.dev/ on 2026-11-13 using its publicly accessible page text.
The turn-taking, live-call examples and prompt-diff review explain concrete user
tasks well. The hero's word “harness” is less concrete than those examples. I would
put a short playable example and a plain-language explanation next to the hero.
Some text-extracted counters read zero; this may be a dynamic-rendering artifact,
so it is not evidence that the browser page displays incorrect production metrics.
No live-call performance or product accuracy was independently tested.
