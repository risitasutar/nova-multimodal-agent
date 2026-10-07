# Re-check of cite-03 after the provenance fix

After the post-verifier smoke run (`../smoke_post_verifier/`) flagged a correct, correctly cited answer, the
claim check was changed to ignore provenance words ("information", "mentioned", "page", "report", ...).
This single-case run, on the fixed code, returned the same answer and verified it: status `VERIFIED`, cited
page 10 (gold page 10).
