% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/alg_elastic_net.R
\name{cla_en}
\alias{cla_en}
\title{Prediction with EN and evaluation through CV - classification}
\usage{
cla_en(
  y,
  X,
  job_name = "new_job",
  K = 10,
  n_pca = NULL,
  pb = FALSE,
  foldset = NULL,
  seed = NULL,
  lasso_TC = trainControl(method = "cv", number = 5, verboseIter = F),
  lasso_TG = NULL,
  n_alpha = 10,
  n_lambda = 100,
  nc = 1
)
}
\arguments{
\item{y}{the responding variable - binary}

\item{X}{the feature matrix. No missing allowed}

\item{job_name}{A character of job ID}

\item{K}{integer, the number of folds in the CV}

\item{n_pca}{number of selected top PCs. Null for no PCA}

\item{pb}{show pgbar or nots}

\item{foldset}{provide a foldset to control random sampling error.}

\item{seed}{provide a seed to reproduce a result. NULL by default}

\item{lasso_TC}{trControl that pass to caret}

\item{lasso_TG}{tuneGrid that pass to caret}

\item{n_alpha}{number of alpha tested in the grid}

\item{n_lambda}{number of lambda tested in the grid}

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
             using EN through CV.
}
