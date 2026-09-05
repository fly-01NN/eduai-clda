# Data sharing and redistribution boundary

## Public in this repository

The release contains the materials needed to inspect and rerun the reported
derived-data analysis:

1. the frozen discovery and classification protocol;
2. collection, parsing, analysis, validation, audit, and plotting code;
3. frozen search rankings and derived project, dependency, model, source-reuse,
   and version-history tables;
4. de-identified independent-review labels and sampling fields;
5. aggregate estimates, bootstrap and permutation draws, robustness results,
   and hard-check records; and
6. author-created figure source data and final figures.

Project, repository, model, revision, and public URL fields remain in the
derived tables where they are needed to audit inclusion, edge provenance, or
source retrieval. These fields refer to public technical artifacts, not study
participants. The human-review release replaces project identifiers and source
excerpts with opaque sample IDs while retaining the labels, inclusion
probabilities, strata, and candidate-boundary indicator needed to reproduce
the reported validation estimates.

## Not redistributed

The package excludes downloaded source-code bodies, raw Hugging Face and
GitHub API responses, local caches, credentials, and private reviewer packages.
Those reviewer packages contain short third-party excerpts and the working
files used to coordinate independent coding. Reviewer identities are not part
of the analytical data.

This boundary avoids repackaging third-party content while preserving the
derived evidence used in the manuscript. It does not imply that the original
public repositories or API records are confidential.

## Independent reconstruction

The released tables reproduce the manuscript's cross-layer distributions,
same-project concentration contrast, project resampling, co-declaration test,
candidate-boundary analyses, source-reuse sensitivities, validation estimates,
and figures without network access. Reconstructing dependency edges directly
from source text requires reacquiring the pinned public revisions listed in the
manifests under the upstream repositories' current terms.

Anyone undertaking a new collection is responsible for checking current
service terms, using an identifiable research user agent, applying conservative
request rates, and recording retrieval dates and revisions. A newly collected
frame is a new dataset rather than a replacement for the frozen snapshot.

