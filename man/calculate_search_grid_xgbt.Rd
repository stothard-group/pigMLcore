% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/alg_xgbtree.R
\name{calculate_search_grid_xgbt}
\alias{calculate_search_grid_xgbt}
\title{calculate the whole grid for xgbtree}
\usage{
calculate_search_grid_xgbt(y, X, n_rounds, n_depth, n_seq)
}
\arguments{
\item{y}{the responding variable. The function assume it is non-missing}

\item{X}{the full feature dataset}

\item{n_rounds}{number of nrounds in grid search,}

\item{n_depth}{number of depth in grid search,}

\item{n_seq}{number of gamma/min_child_weight in grid search,}
}
\value{
a data frame
}
\description{
DESCRIPTION
}
\note{
A search grid calculated based on the input data
}
