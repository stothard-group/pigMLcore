% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/alg_SVR.R
\name{cla_svrr}
\alias{cla_svrr}
\title{Prediction with SVR polynomialand evaluation through CV}
\usage{
cla_svrr(
  y,
  X,
  job_name = "new_job",
  K = 10,
  n_pca = NULL,
  pb = FALSE,
  foldset = NULL,
  seed = NULL,
  svrr_TC = trainControl(method = "cv", number = 5, verboseIter = F),
  svrr_TG = NULL,
  n_C = 10,
  nc = 1
)
}
\arguments{
\item{y}{the responding variable}

\item{X}{the feature matrix. No missing allowed}

\item{job_name}{A character of job ID}

\item{K}{integer, the number of folds in the CV}

\item{n_pca}{number of selected top PCs. Null for no PCA}

\item{pb}{show pgbar or nots}

\item{foldset}{provide a foldset to control random sampling error.}

\item{seed}{provide a seed to reproduce a result. NULL by default}

\item{svrr_TC}{trControl that pass to caret}

\item{svrr_TG}{tuneGrid that pass to caret}

\item{n_C}{number of Cost in grid search}

\item{nc}{number of cpus to use in parallel}
}
\value{
a list
}
\description{
DESCRIPTION
}
\note{
The function provide a general way to evaluate the prediction
             using SVR RBF through CV.
}
