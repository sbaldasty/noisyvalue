.noisyvalue_env <- new.env(parent = emptyenv())

#' Point reticulate at the noisyvalue Python package
#'
#' Call this once per R session, before any other function in this package.
#' The package is a thin interface to the Python `noisyvalue` library, which
#' does the posterior computation; this selects the Python environment that
#' holds it.
#'
#' @param venv Path to the Python virtualenv (defaults to `.venv` in the
#'   current working directory, i.e. the repo root).
#' @param src Path to the `src` directory containing the Python library.
#' @export
noisyvalue_init <- function(venv = ".venv", src = "src") {
  reticulate::use_virtualenv(normalizePath(venv), required = TRUE)

  sys <- reticulate::import("sys")
  src <- normalizePath(src)
  if (!(src %in% sys$path)) {
    sys$path <- c(sys$path, src)
  }

  .noisyvalue_env$builtins <- reticulate::import_builtins(convert = FALSE)
  .noisyvalue_env$pd <- reticulate::import("pandas", convert = FALSE)
  .noisyvalue_env$np <- reticulate::import("numpy", convert = FALSE)
  .noisyvalue_env$core <- reticulate::import("noisyvalue.core", convert = FALSE)
  .noisyvalue_env$io <- reticulate::import("noisyvalue.io", convert = FALSE)
  .noisyvalue_env$pdext <- reticulate::import("noisyvalue.pandas", convert = FALSE)
  .noisyvalue_env$census <- reticulate::import("noisyvalue.census", convert = FALSE)

  invisible(TRUE)
}

.nv <- function(name) {
  val <- .noisyvalue_env[[name]]
  if (is.null(val)) {
    stop("noisyvalue_init() must be called before using this function", call. = FALSE)
  }
  val
}
