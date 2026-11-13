# Independent answer review

Review `blind_answers.json` against the original PDFs named in `packet_manifest.json`. It omits architecture, pipeline status, and verifier verdict. Do not treat cited excerpts alone as an answer key. Check numerical values, qualifications, all question parts, and whether each cited page supports its attached claims. Check every required paper in comparisons.

Enter grades in `grades.csv`: accuracy=1 only for a fully correct, complete answer; citation_grounding=1 only when all factual claims have supporting citations. A justified refusal on an unanswerable question can receive 1 for both; an inappropriate refusal on an answerable question receives 0 for both. For U questions, also enter correct_refusal=0 or 1; leave that cell blank for other questions. Record your reviewer name, original PDF page numbers, and a specific rationale. Partial correctness receives accuracy=0 under this binary rubric; explain the correct parts.

Infrastructure failures are excluded from answer grades and remain reliability outcomes in `infrastructure_failures.csv`. Packet refreshes preserve existing request-keyed grades. No grade has been generated automatically.
