# Frozen split manifests

The model-facing manifests preserve the official FLEURS boundaries and contain
1,902 train, 210 validation, and 252 test clips. FLEURS documents those source
boundaries as speaker-disjoint. This TSV release has no speaker ID column, so
the repository cannot independently count speakers or prove disjointness by
identifier; see `../README.md` for the exact scope.
