#!/usr/bin/env bash
# Builds the R package documentation site into r/noisyvalue/docs.
# Usage: r/build_docs.sh [--serve]
set -euo pipefail

pkg_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/noisyvalue" && pwd)"
repo_root="$(dirname "$(dirname "$pkg_dir")")"

# The R package drives the Python library through reticulate.
if [[ -z "${RETICULATE_PYTHON:-}" && -x "$repo_root/.venv/bin/python" ]]; then
  export RETICULATE_PYTHON="$repo_root/.venv/bin/python"
fi

cd "$pkg_dir"
Rscript -e '
for (p in c("pkgdown", "roxygen2", "devtools", "rmarkdown", "knitr")) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org")
  }
}
roxygen2::roxygenise()
pkgdown::build_site(preview = FALSE, install = TRUE, new_process = FALSE)
'

echo "Built: $pkg_dir/docs/index.html"

if [[ "${1:-}" == "--serve" ]]; then
  echo "Serving on http://localhost:8000"
  python3 -m http.server 8000 --directory "$pkg_dir/docs"
fi
