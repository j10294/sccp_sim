suppressPackageStartupMessages({
  library(reticulate)
  library(dplyr)
  library(purrr)
  library(tibble)
})

# ----------------------------
# Utilities
# ----------------------------
conformal_quantile <- function(scores, alpha) {
  scores <- sort(as.numeric(scores))
  n <- length(scores)
  if (n == 0) return(Inf)
  k <- ceiling((n + 1) * (1 - alpha))
  k <- max(min(k, n), 1)
  scores[k]
}

# Base metrics (overall)
eval_metrics <- function(pred_sets, y_true, K) {
  covered <- map2_lgl(pred_sets, y_true, ~ (.y %in% .x))
  tibble(
    overall_cov = mean(covered),
    mean_set_size = mean(lengths(pred_sets)),
    median_set_size = median(lengths(pred_sets))
  )
}

# ----------------------------
# Tail-focused + classwise metrics 
# ----------------------------

# Define tail classes using a reference label sample (recommend: y_cal)
# tail_frac = proportion of rare classes (e.g., 0.2 => bottom 20% by frequency)
get_tail_classes <- function(y_ref, K, tail_frac = 0.2) {
  y_ref <- as.integer(y_ref)
  tab <- tabulate(y_ref, nbins = K)
  cls <- seq_len(K)
  ord <- order(tab, decreasing = FALSE) # rare -> frequent
  m <- max(1, ceiling(K * tail_frac))
  tail_cls <- cls[ord[1:m]]
  list(tail_cls = tail_cls, freq = tab)
}

# Compute tail-only metrics given pred_sets + y_true + tail_cls
eval_metrics_tail <- function(pred_sets, y_true, tail_cls) {
  y_true <- as.integer(y_true)
  is_tail <- y_true %in% tail_cls
  if (!any(is_tail)) {
    return(tibble(
      overall_cov_tail = NA_real_,
      mean_set_size_tail = NA_real_,
      median_set_size_tail = NA_real_,
      n_tail = 0L
    ))
  }

  covered <- map2_lgl(pred_sets, y_true, ~ (.y %in% .x))
  ss <- lengths(pred_sets)

  tibble(
    overall_cov_tail = mean(covered[is_tail]),
    mean_set_size_tail = mean(ss[is_tail]),
    median_set_size_tail = median(ss[is_tail]),
    n_tail = sum(is_tail)
  )
}

# Classwise coverage table
eval_classwise_cov <- function(pred_sets, y_true, K) {
  y_true <- as.integer(y_true)
  covered <- map2_lgl(pred_sets, y_true, ~ (.y %in% .x))

  # Ensure all classes 1..K are present
  cov_by_class <- rep(NA_real_, K)
  n_by_class   <- tabulate(y_true, nbins = K)

  for (k in seq_len(K)) {
    idx <- which(y_true == k)
    if (length(idx) > 0) cov_by_class[k] <- mean(covered[idx])
  }

  tibble(
    y = seq_len(K),
    n = as.integer(n_by_class),
    cov = as.numeric(cov_by_class)
  )
}

# Summaries that often show SCCP gains
summarize_classwise <- function(df_classwise, tail_cls) {
  df_tail <- df_classwise %>% filter(.data$y %in% tail_cls, .data$n > 0)
  df_all  <- df_classwise %>% filter(.data$n > 0)

  tibble(
    worst_class_cov = if (nrow(df_all) > 0) min(df_all$cov, na.rm = TRUE) else NA_real_,
    var_class_cov   = if (nrow(df_all) > 1) var(df_all$cov, na.rm = TRUE) else NA_real_,
    tail_mean_cov   = if (nrow(df_tail) > 0) mean(df_tail$cov, na.rm = TRUE) else NA_real_,
    tail_worst_cov  = if (nrow(df_tail) > 0) min(df_tail$cov, na.rm = TRUE) else NA_real_
  )
}

# ----------------------------
# Clusterwise metrics (NEW)
# label_clusters: integer vector length K (label -> cluster id in 1..Kc_eff)
# ----------------------------

eval_clusterwise_metrics <- function(pred_sets, y_true, label_clusters, Kc_eff = NULL) {
  y_true <- as.integer(y_true)
  cl_true <- label_clusters[y_true]

  if (is.null(Kc_eff)) Kc_eff <- max(label_clusters)

  covered <- map2_lgl(pred_sets, y_true, ~ (.y %in% .x))
  ss <- lengths(pred_sets)

  cov_by_cluster <- rep(NA_real_, Kc_eff)
  size_by_cluster <- rep(NA_real_, Kc_eff)
  n_by_cluster <- tabulate(cl_true, nbins = Kc_eff)

  for (cc in seq_len(Kc_eff)) {
    idx <- which(cl_true == cc)
    if (length(idx) > 0) {
      cov_by_cluster[cc]  <- mean(covered[idx])
      size_by_cluster[cc] <- mean(ss[idx])
    }
  }

  tibble(
    cluster = seq_len(Kc_eff),
    n = as.integer(n_by_cluster),
    cov = as.numeric(cov_by_cluster),
    mean_set_size = as.numeric(size_by_cluster)
  )
}

summarize_clusterwise <- function(df_clusterwise) {
  df_ok <- df_clusterwise %>% filter(.data$n > 0)

  tibble(
    worst_cluster_cov = if (nrow(df_ok) > 0) min(df_ok$cov, na.rm = TRUE) else NA_real_,
    var_cluster_cov   = if (nrow(df_ok) > 1) var(df_ok$cov, na.rm = TRUE) else NA_real_,
    mean_cluster_size = if (nrow(df_ok) > 0) mean(df_ok$mean_set_size, na.rm = TRUE) else NA_real_,
    max_cluster_size  = if (nrow(df_ok) > 0) max(df_ok$mean_set_size, na.rm = TRUE) else NA_real_
  )
}

# ----------------------------
# Prediction set constructors
# ----------------------------
build_pred_set_global <- function(p_row, qG) {
  labels <- as.integer(names(p_row))
  s_vec  <- 1 - p_row
  labels[s_vec <= qG]
}

build_pred_set_cluster <- function(p_row, label_clusters, q_by_cluster) {
  labels <- as.integer(names(p_row))
  s_vec  <- 1 - p_row
  cl_ids <- label_clusters[labels]
  th_vec <- q_by_cluster[cl_ids]
  labels[s_vec <= th_vec]
}

# ----------------------------
# Label clustering (k-means on score embeddings)
# scores_by_label: list, index by label (1..K), values are scores for that label
# ----------------------------
label_kmeans <- function(scores_by_label,
                         Kc = 10,
                         alpha = 0.05,
                         quantiles = c(.5, .6, .7, .8, .9, 1 - alpha)) {

  quantiles <- sort(unique(quantiles))
  K <- length(scores_by_label)
  emb <- matrix(NA_real_, nrow = K, ncol = length(quantiles))

  for (k in seq_len(K)) {
    v <- scores_by_label[[as.character(k)]]
    if (is.null(v)) v <- numeric(0)
    if (length(v) < 5) {
      fill <- if (length(v) == 0) 0.5 else median(v, na.rm = TRUE)
      v <- c(v, rep(fill, 5 - length(v)))
    }
    emb[k, ] <- quantile(v, probs = quantiles, na.rm = TRUE, type = 8)
  }

  Kc_eff <- min(Kc, K)
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
  for (c in small) lab[lab == c] <- big
  uniq <- sort(unique(lab))
  match(lab, uniq)
}

# ----------------------------
# Lambda selection on selection split (cluster-wise)
# qC_sel: length Kc_eff
# qG_sel: scalar
# ----------------------------
choose_lambda_by_selection <- function(
    p_sel, y_sel, label_clusters_sel,
    alpha, qC_sel, qG_sel,
    grid = seq(0, 1, by = 0.1)
) {
  Kc_eff <- length(qC_sel)
  best_lambda <- rep(NA_real_, Kc_eff)

  cand_lmb <- sort(unique(grid))

  cand <- lapply(cand_lmb, function(lmb) {
    q_star <- (1 - lmb) * qC_sel + lmb * qG_sel
    pred_sets <- lapply(seq_len(nrow(p_sel)), function(i) {
      pr <- p_sel[i, ]; names(pr) <- colnames(p_sel)
      build_pred_set_cluster(pr, label_clusters_sel, q_star)
    })
    covered <- mapply(function(ps, y) y %in% ps, pred_sets, y_sel)
    list(pred_sets = pred_sets, covered = covered)
  })

  mean_size_vec <- sapply(cand, function(o) mean(lengths(o$pred_sets)))
  cov_overall_vec <- sapply(cand, function(o) mean(o$covered))

  for (cc in seq_len(Kc_eff)) {
    idx_cc <- which(label_clusters_sel[y_sel] == cc)

    if (length(idx_cc) == 0) {
      best_lambda[cc] <- 1
      next
    }

    cov_cc_vec <- sapply(cand, function(o) mean(o$covered[idx_cc]))

    eval_tbl <- tibble(
      lambda      = cand_lmb,
      mean_size   = mean_size_vec,
      cov_cc      = cov_cc_vec,
      cov_overall = cov_overall_vec
    )

    feas <- eval_tbl %>% filter(cov_cc >= (1 - alpha), cov_overall >= (1 - alpha))

    if (nrow(feas) > 0) {
      chosen <- feas %>% arrange(mean_size, desc(cov_cc), desc(cov_overall)) %>% slice(1)
    } else {
      chosen <- eval_tbl %>%
        mutate(
          viol_cc      = pmax(0, (1 - alpha) - cov_cc),
          viol_overall = pmax(0, (1 - alpha) - cov_overall),
          loss         = viol_cc + viol_overall + mean_size
        ) %>%
        arrange(loss, mean_size) %>%
        slice(1)
    }

    best_lambda[cc] <- chosen$lambda
  }

  best_lambda[is.na(best_lambda)] <- 1
  best_lambda
}

# ----------------------------
# Main: Run GCP / CCCP / SCCP from NPZ
# ----------------------------
run_cifar_from_npz <- function(npz_path,
                               alpha = 0.05,
                               Kc = 10,
                               m_min = 100,
                               lambda_grid = seq(0, 1, by = 0.1),
                               tail_frac = 0.2) {

  np <- reticulate::import("numpy", delay_load = TRUE)
  z  <- np$load(npz_path, allow_pickle = TRUE)

  p_sel <- as.matrix(z[["p_sel"]])
  p_cal <- as.matrix(z[["p_cal"]])
  p_tst <- as.matrix(z[["p_tst"]])

  y_sel <- as.integer(z[["y_sel"]]) + 1L
  y_cal <- as.integer(z[["y_cal"]]) + 1L
  y_tst <- as.integer(z[["y_tst"]]) + 1L

  K <- ncol(p_cal)
  colnames(p_sel) <- colnames(p_cal) <- colnames(p_tst) <- as.character(seq_len(K))

  get_true_prob <- function(p_mat, y) {
    idx <- match(as.character(y), colnames(p_mat))
    p_mat[cbind(seq_len(nrow(p_mat)), idx)]
  }

  s_sel_true <- 1 - get_true_prob(p_sel, y_sel)
  s_cal_true <- 1 - get_true_prob(p_cal, y_cal)

  # Tail classes defined from calibration labels (recommended; avoids using test)
  tail_info <- get_tail_classes(y_ref = y_cal, K = K, tail_frac = tail_frac)
  tail_cls <- tail_info$tail_cls

  # ---- GCP
  qG_cal <- conformal_quantile(s_cal_true, alpha)
  pred_GCP <- lapply(seq_len(nrow(p_tst)), function(i) {
    pr <- p_tst[i, ]; names(pr) <- colnames(p_tst)
    build_pred_set_global(pr, qG_cal)
  })
  met_GCP_overall <- eval_metrics(pred_GCP, y_tst, K)
  met_GCP_tail    <- eval_metrics_tail(pred_GCP, y_tst, tail_cls)

  cw_GCP <- eval_classwise_cov(pred_GCP, y_tst, K)
  sum_GCP <- summarize_classwise(cw_GCP, tail_cls)

  clw_GCP  <- eval_clusterwise_metrics(pred_GCP, y_tst, label_clusters, Kc_eff = Kc_eff)
  clsum_GCP <- summarize_clusterwise(clw_GCP)

  


  # ---- CC-CP
  scores_by_label_cal <- split(s_cal_true, y_cal)
  scores_by_label_cal <- scores_by_label_cal[as.character(seq_len(K))]

  label_clusters <- label_kmeans(scores_by_label_cal, Kc = Kc, alpha = alpha,
                                 quantiles = c(.5, .6, .7, .8, .9, 1 - alpha))
  label_clusters <- merge_small_clusters(label_clusters, y_cal, m_min = m_min)
  Kc_eff <- length(unique(label_clusters))

  qC_cal <- sapply(seq_len(Kc_eff), function(cc) {
    idx <- which(label_clusters[y_cal] == cc)
    conformal_quantile(s_cal_true[idx], alpha)
  })

  pred_CCCP <- lapply(seq_len(nrow(p_tst)), function(i) {
    pr <- p_tst[i, ]; names(pr) <- colnames(p_tst)
    build_pred_set_cluster(pr, label_clusters, qC_cal)
  })
  met_CCCP_overall <- eval_metrics(pred_CCCP, y_tst, K)
  met_CCCP_tail    <- eval_metrics_tail(pred_CCCP, y_tst, tail_cls)

  cw_CCCP <- eval_classwise_cov(pred_CCCP, y_tst, K)
  sum_CCCP <- summarize_classwise(cw_CCCP, tail_cls)

  clw_CCCP  <- eval_clusterwise_metrics(pred_CCCP, y_tst, label_clusters, Kc_eff = Kc_eff)
  clsum_CCCP <- summarize_clusterwise(clw_CCCP)


  # ---- SCC-CP
  qC_sel <- sapply(seq_len(Kc_eff), function(cc) {
    idx <- which(label_clusters[y_sel] == cc)
    conformal_quantile(s_sel_true[idx], alpha)
  })
  qG_sel <- conformal_quantile(s_sel_true, alpha)

  lambda_hat <- choose_lambda_by_selection(
    p_sel = p_sel, y_sel = y_sel,
    label_clusters_sel = label_clusters,
    alpha = alpha,
    qC_sel = qC_sel, qG_sel = qG_sel,
    grid = lambda_grid
  )

  q_star <- (1 - lambda_hat) * qC_cal + lambda_hat * qG_cal

  pred_SCCP <- lapply(seq_len(nrow(p_tst)), function(i) {
    pr <- p_tst[i, ]; names(pr) <- colnames(p_tst)
    build_pred_set_cluster(pr, label_clusters, q_star)
  })
  met_SCCP_overall <- eval_metrics(pred_SCCP, y_tst, K)
  met_SCCP_tail    <- eval_metrics_tail(pred_SCCP, y_tst, tail_cls)

  cw_SCCP <- eval_classwise_cov(pred_SCCP, y_tst, K)
  sum_SCCP <- summarize_classwise(cw_SCCP, tail_cls)

  clw_SCCP  <- eval_clusterwise_metrics(pred_SCCP, y_tst, label_clusters, Kc_eff = Kc_eff)
  clsum_SCCP <- summarize_clusterwise(clw_SCCP)


  # ---- Output tables
  overall_tbl <- bind_rows(
    met_GCP_overall  %>% mutate(method = "GCP"),
    met_CCCP_overall %>% mutate(method = "CCCP"),
    met_SCCP_overall %>% mutate(method = "SCCP")
  )

  tail_tbl <- bind_rows(
    met_GCP_tail  %>% mutate(method = "GCP"),
    met_CCCP_tail %>% mutate(method = "CCCP"),
    met_SCCP_tail %>% mutate(method = "SCCP")
  )
  classwise_summary_tbl <- bind_rows(
    sum_GCP  %>% mutate(method = "GCP"),
    sum_CCCP %>% mutate(method = "CCCP"),
    sum_SCCP %>% mutate(method = "SCCP")
  )

  clusterwise_summary_tbl = bind_rows(
  clsum_GCP  %>% mutate(method = "GCP"),
  clsum_CCCP %>% mutate(method = "CCCP"),
  clsum_SCCP %>% mutate(method = "SCCP")
  )


  list(
    overall = overall_tbl,
    tail = tail_tbl,
    classwise_summary = classwise_summary_tbl,
    clusterwise_summary = clusterwise_summary_tbl,
    classwise = list(GCP = cw_GCP, CCCP = cw_CCCP, SCCP = cw_SCCP),
    clusterwise = list(GCP = clw_GCP, CCCP = clw_CCCP, SCCP = clw_SCCP),
    tail_info = list(tail_frac = tail_frac, tail_classes = tail_cls, freq_cal = tail_info$freq),
    label_clusters = label_clusters,
    lambda_hat = lambda_hat,
    qG_cal = qG_cal,
    qC_cal = qC_cal,
    q_star = q_star
  )
}
