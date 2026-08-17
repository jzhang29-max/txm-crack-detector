# Background documents

These record how the model was arrived at. None of it is needed to use the app --
start with the README in the parent directory.

- **SAM_COMPARISON.md** — the full study behind the model choice: zero-shot SAM
  measured against the deployed classifier, with 33 verified citations. Answers
  "why not just use Segment Anything?"
- **RESEARCH_NOTES.md** — the development record, including four approaches that
  were adopted and then reverted as regressions (flat-fielding as model input,
  geometric masking, a curvilinearity gate, algorithmic crack labels). Kept so
  they are not retried, and because the reasoning behind the metric rules lives
  here: an over-aggressive filter and a good one both reduce predicted area, and
  only recall against ground truth separates them.
- **APP_COMPARISON.md** — how this app compares to the sibling SEM pipeline's,
  what was copied in each direction, and the shared layout both now use.
