# =========================================================
# R/methods.R
# - Packages
# - Utilities, metrics, predset builders
# - Label clustering + lambda selection
# - Data generator
# - Method runners: GCP / CCCP / SCCCP
# =========================================================

suppressPackageStartupMessages({
  library(nnet)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(MASS)
  library(progress)
  library(scales)
  library(tibble)
  library(xtable)
})

# ----------------------------
# 1) Basic utilities
# ----------------------------

conformal_quantile <- function(scores, alpha) {
  scores <- sort(as.numeric(scores))
  n <- length(scores)
  if (n == 0) return(Inf)
  k <- ceiling((n + 1) * (1 - alpha))
  k <- max(min(k, n), 1)
  scores[k]
}

ensure_dirs <- function(paths) {
  for (p in paths) if (!dir.exists(p)) dir.create(p, recursive = TRUE)
  invisible(TRUE)
}

# Extract true label probabilities from probability matrix
get_true_prob <- function(p_mat, y) {
  y_chr   <- as.character(y)
  col_idx <- match(y_chr, colnames(p_mat))
  if (any(is.na(col_idx))) {
    missing_labels <- unique(y_chr[is.na(col_idx)])
    stop("Some labels in y are not found in colnames(p_mat): ",
         paste(missing_labels, collapse = ", "))
  }
  p_mat[cbind(seq_len(nrow(p_mat)), col_idx)]
}

# ----------------------------
# 2) Metrics
# ----------------------------

.safe_cov_variance <- function(per_class_cov, overall_cov, n_k) {
  ok <- is.finite(per_class_cov) & (n_k > 0)
  if (!any(ok)) return(NA_real_)
  mean((per_class_cov[ok] - overall_cov)^2)
}

eval_metrics <- function(pred_sets, y_true, K) {
  covered <- map2_lgl(pred_sets, y_true, ~ (.y %in% .x))
  overall_cov <- mean(covered)

  set_sizes <- vapply(pred_sets, length, integer(1))
  mean_set_size   <- mean(set_sizes)
  median_set_size <- median(set_sizes)

  n_k <- tabulate(y_true, nbins = K)
  per_class_cov <- vapply(seq_len(K), function(k) {
    if (n_k[k] == 0) return(NA_real_)
    mean(covered[y_true == k])
  }, numeric(1))
  per_class <- tibble::tibble(
    class = seq_len(K),
    n_k   = n_k,
    cov   = per_class_cov
  )

  coverage_variance <- .safe_cov_variance(per_class$cov, overall_cov, per_class$n_k)
  worst_class_cov <- if (any(per_class$n_k > 0)) {
    min(per_class$cov[per_class$n_k > 0], na.rm = TRUE)
  } else {
    NA_real_
  }

  tibble::tibble(
    overall_cov = overall_cov,
    mean_set_size = mean_set_size,
    median_set_size = median_set_size,
    cov_var_across_classes = coverage_variance,
    worst_class_cov = worst_class_cov,
    coverage_variance = coverage_variance
  ) %>% tibble::add_column(per_class = list(per_class))
}

compute_classwise <- function(predsets, true_y, K) {
  out <- lapply(seq_len(K), function(k){
    idx <- which(true_y == k)
    if (length(idx) == 0) {
      return(data.frame(class = k,
                        class_cov = NA_real_,
                        class_size = NA_real_))
    }
    cov_k <- mean(sapply(idx, function(i) k %in% predsets[[i]]))
    size_k <- mean(sapply(idx, function(i) length(predsets[[i]])))
    data.frame(class = k,
               class_cov = cov_k,
               class_size = size_k)
  })
  do.call(rbind, out)
}

# ----------------------------
# 3) Prediction set builders
# ----------------------------

build_pred_set_global <- function(p_row_named, qG) {
  labels <- as.integer(names(p_row_named))
  s_vec  <- 1 - p_row_named
  labels[s_vec <= qG]
}

build_pred_set_cluster <- function(p_row_named, label_clusters, q_by_cluster) {
  labels <- as.integer(names(p_row_named))
  s_vec  <- 1 - p_row_named
  cl_ids <- label_clusters[labels]
  th_vec <- q_by_cluster[cl_ids]
  labels[s_vec <= th_vec]
}

# Predset maker helpers (reduce repeated lapply blocks)
make_predsets_global <- function(p_tst, qG) {
  lapply(seq_len(nrow(p_tst)), function(i) {
    pr <- p_tst[i, ]
    pr_named <- pr; names(pr_named) <- colnames(p_tst)
    build_pred_set_global(pr_named, qG)
  })
}

make_predsets_cluster <- function(p_tst, label_clusters, q_vec) {
  lapply(seq_len(nrow(p_tst)), function(i) {
    pr <- p_tst[i, ]
    pr_named <- pr; names(pr_named) <- colnames(p_tst)
    build_pred_set_cluster(pr_named, label_clusters, q_vec)
  })
}

# ----------------------------
# 4) Label clustering
# ----------------------------

label_kmeans <- function(scores_by_label,
                         Kc = 5,
                         alpha = 0.05,
                         quantiles = c(.1, .3, .5, .7, .9, 1 - alpha)) {
  quantiles <- sort(unique(quantiles))
  K <- length(scores_by_label)
  emb <- matrix(NA_real_, nrow = K, ncol = length(quantiles))

  for (k in seq_len(K)) {
    v <- scores_by_label[[k]]
    if (length(v) < 5) {
      v <- c(v, rep(median(v, na.rm = TRUE), 5 - length(v)))
    }
    emb[k, ] <- quantile(v, probs = quantiles, na.rm = TRUE, type = 8)
  }

  Kc_eff <- min(Kc, K)
  if (Kc_eff < 1) stop("label_kmeans: effective Kc < 1.")

  km <- kmeans(scale(emb), centers = Kc_eff, nstart = 10)
  km$cluster
}

merge_small_clusters <- function(label_clusters, y_cal, m_min = 100) {
  cl_idx <- label_clusters[y_cal]
  cl_counts <- tabulate(cl_idx, nbins = max(label_clusters))
  small <- which(cl_counts < m_min)
  if (length(small) == 0) return(label_clusters)

  big <- which.max(cl_counts)
  lab <- label_clusters
  for (c in small) {
    lab[lab == c] <- big
  }
  uniq <- sort(unique(lab))
  match(lab, uniq)
}

# ----------------------------
# 5) Lambda selection (SCCCP)
# ----------------------------

choose_lambda_by_selection <- function(
    p_sel,
    y_sel,
    label_clusters_sel,
    alpha,
    qC_sel,
    qG_sel,
    grid = seq(0, 1, by = 0.1)
) {
  Kc <- length(unique(label_clusters_sel))
  best_lambda <- rep(NA_real_, Kc)

  make_eval <- function(lmb) {
    q_star <- (1 - lmb) * qC_sel + lmb * qG_sel
    pred_sets <- make_predsets_cluster(p_sel, label_clusters_sel, q_star)
    covered <- mapply(function(ps, y) y %in% ps, pred_sets, y_sel)
    list(pred_sets = pred_sets, covered = covered)
  }

  cand_lmb <- sort(unique(grid))
  cand     <- lapply(cand_lmb, make_eval)

  mean_size_vec <- sapply(cand, function(o) {
    mean(sapply(o$pred_sets, length))
  })

  cov_overall_vec <- sapply(cand, function(o) mean(o$covered))

  for (cc in seq_len(Kc)) {
    idx_cc <- which(label_clusters_sel[y_sel] == cc)

    if (length(idx_cc) == 0) {
      best_lambda[cc] <- 1
      next
    }

    cov_cc_vec <- sapply(cand, function(o) mean(o$covered[idx_cc]))

    eval_tbl <- tibble::tibble(
      lambda       = cand_lmb,
      mean_size    = mean_size_vec,
      cov_cc       = cov_cc_vec,
      cov_overall  = cov_overall_vec
    )

    feas <- dplyr::filter(
      eval_tbl,
      cov_cc      >= (1 - alpha),
      cov_overall >= (1 - alpha)
    )

    if (nrow(feas) > 0) {
      chosen <- feas %>%
        dplyr::arrange(mean_size,
                       dplyr::desc(cov_cc),
                       dplyr::desc(cov_overall)) %>%
        dplyr::slice(1)
    } else {
      eval_tbl <- eval_tbl %>%
        dplyr::mutate(
          viol_cc      = pmax(0, (1 - alpha) - cov_cc),
          viol_overall = pmax(0, (1 - alpha) - cov_overall),
          loss         = viol_cc + viol_overall + mean_size
        )
      chosen <- eval_tbl %>%
        dplyr::arrange(loss, mean_size) %>%
        dplyr::slice(1)
    }

    best_lambda[cc] <- chosen$lambda
  }

  best_lambda[is.na(best_lambda)] <- 1
  best_lambda
}

# ----------------------------
# 6) Data generator
# ----------------------------

gen_data <- function(n, K = 50, d = 10,
                     prior = c("balanced","zipf"),
                     noise = c("homog","hetero"),
                     sep = 1.8) {
  prior <- match.arg(prior)
  noise <- match.arg(noise)

  if (prior == "balanced") {
    pi_k <- rep(1 / K, K)
  } else {
    pi_k <- 1 / (1:K)^1.1
    pi_k <- pi_k / sum(pi_k)
  }

  base_mus <- matrix(rnorm(K * d), K, d)
  base_mus <- sweep(base_mus, 1, sqrt(rowSums(base_mus^2)), "/")

  mus    <- matrix(NA_real_, K, d)
  sigmas <- vector("list", K)

  if (noise == "homog") {
    for (k in 1:K) {
      mus[k, ]    <- sep * base_mus[k, ]
      sigmas[[k]] <- diag(d)
    }
  } else if (noise == "hetero") {
    for (k in 1:K) {
      mus[k, ] <- sep * base_mus[k, ]
      s        <- runif(1, 0.5, 3.0)
      sigmas[[k]] <- diag(d) * s
    }
  }

  y <- sample(seq_len(K), size = n, replace = TRUE, prob = pi_k)
  X <- matrix(NA_real_, n, d)
  for (i in seq_len(n)) {
    k <- y[i]
    X[i, ] <- MASS::mvrnorm(1, mu = mus[k, ], Sigma = sigmas[[k]])
  }

  list(X = X, y = y)
}

# ----------------------------
# 7) Method runners
# ----------------------------

fit_GCP <- function(p_cal, y_cal, p_tst, y_te, K, alpha) {
  s_cal_true <- 1 - get_true_prob(p_cal, y_cal)
  qG_cal <- conformal_quantile(s_cal_true, alpha)

  predsets <- make_predsets_global(p_tst, qG_cal)

  list(
    predsets  = predsets,
    metrics   = eval_metrics(predsets, y_te, K),
    classwise = compute_classwise(predsets, y_te, K),
    qG_cal    = qG_cal,
    s_cal_true = s_cal_true
  )
}

fit_CCCP <- function(s_cal_true, y_cal, p_tst, y_te, K, alpha, Kc,
                     m_min = 100) {
  scores_by_label_cal <- split(s_cal_true, y_cal)

  label_clusters <- label_kmeans(scores_by_label_cal, Kc = Kc,
                                 alpha = alpha,
                                 quantiles = c(.1,.3,.5,.7,.9,1-alpha))
  label_clusters <- merge_small_clusters(label_clusters, y_cal, m_min = m_min)
  Kc_eff <- length(unique(label_clusters))

  qC_cal <- sapply(seq_len(Kc_eff), function(cc) {
    idx <- which(label_clusters[y_cal] == cc)
    conformal_quantile(s_cal_true[idx], alpha)
  })

  size_cc <- sapply(seq_len(Kc_eff), function(cc) sum(label_clusters[y_cal] == cc))
  tau <- 0.0 + 0.05 * as.numeric(size_cc < 150) + 0.03 * as.numeric(size_cc < 80)
  qC_cal_safe <- qC_cal + tau

  predsets <- make_predsets_cluster(p_tst, label_clusters, qC_cal_safe)

  list(
    predsets = predsets,
    metrics = eval_metrics(predsets, y_te, K),
    classwise = compute_classwise(predsets, y_te, K),
    label_clusters = label_clusters,
    qC_cal = qC_cal_safe,
    Kc_eff = Kc_eff
  )
}

fit_SCCCP <- function(p_sel, y_sel, s_sel_true,
                      p_tst, y_te, K, alpha,
                      label_clusters, qC_cal_safe, qG_cal,
                      grid = seq(0, 1, by = 0.1)) {

  Kc_eff <- length(unique(label_clusters))

  qC_sel <- sapply(seq_len(Kc_eff), function(cc) {
    idx <- which(label_clusters[y_sel] == cc)
    conformal_quantile(s_sel_true[idx], alpha)
  })
  qG_sel <- conformal_quantile(s_sel_true, alpha)

  lambda_hat <- choose_lambda_by_selection(
    p_sel              = p_sel,
    y_sel              = y_sel,             
    label_clusters_sel = label_clusters,
    alpha              = alpha,
    qC_sel             = qC_sel,
    qG_sel             = qG_sel,
    grid               = grid
  )

  q_star <- (1 - lambda_hat) * qC_cal_safe + lambda_hat * qG_cal

  predsets <- make_predsets_cluster(p_tst, label_clusters, q_star)

  list(
    predsets = predsets,
    metrics = eval_metrics(predsets, y_te, K),
    classwise = compute_classwise(predsets, y_te, K),
    lambda_hat = lambda_hat,
    q_star = q_star,
    qC_sel = qC_sel,
    qG_sel = qG_sel
  )
}

