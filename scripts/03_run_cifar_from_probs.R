load_cifar_npz <- function(npz_path){
  suppressPackageStartupMessages(library(reticulate))
  np <- reticulate::import("numpy", delay_load = TRUE)
  z  <- np$load(npz_path, allow_pickle = TRUE)
  
  p_sel <- z[["p_sel"]]
  y_sel <- z[["y_sel"]]
  p_cal <- z[["p_cal"]]
  y_cal <- z[["y_cal"]]
  p_tst <- z[["p_tst"]]
  y_tst <- z[["y_tst"]]
  
  # convert
  p_sel <- as.matrix(p_sel)
  p_cal <- as.matrix(p_cal)
  p_tst <- as.matrix(p_tst)
  
  y_sel <- as.integer(y_sel) + 1L
  y_cal <- as.integer(y_cal) + 1L
  y_tst <- as.integer(y_tst) + 1L
  
  K <- ncol(p_sel)
  stopifnot(ncol(p_cal) == K, ncol(p_tst) == K)
  
  # IMPORTANT: give class labels 1..K as colnames (your build_pred_set_* uses names)
  colnames(p_sel) <- as.character(seq_len(K))
  colnames(p_cal) <- as.character(seq_len(K))
  colnames(p_tst) <- as.character(seq_len(K))
  
  list(
    p_sel = p_sel, y_sel = y_sel,
    p_cal = p_cal, y_cal = y_cal,
    p_tst = p_tst, y_tst = y_tst,
    K = K
  )
}

 ----------------------------
# Run from npz file
## ----------------------------

run_once_from_npz <- function(npz_path, alpha = 0.05, Kc = 10){
  dat <- load_cifar_npz(npz_path)
  p_sel <- dat$p_sel; y_sel <- dat$y_sel
  p_cal <- dat$p_cal; y_cal <- dat$y_cal
  p_tst <- dat$p_tst; y_te  <- dat$y_tst
  K <- dat$K
  
  # true-label scores
  get_true_prob <- function(p_mat, y){
    col_idx <- match(as.character(y), colnames(p_mat))
    p_mat[cbind(seq_len(nrow(p_mat)), col_idx)]
  }
  s_sel_true <- 1 - get_true_prob(p_sel, y_sel)
  s_cal_true <- 1 - get_true_prob(p_cal, y_cal)
  
  # ----- GCP -----
  qG_cal <- conformal_quantile(s_cal_true, alpha)
  predsets_GCP <- lapply(seq_len(nrow(p_tst)), function(i){
    pr <- p_tst[i, ]; names(pr) <- colnames(p_tst)
    build_pred_set_global(pr, qG_cal)
  })
  metrics_GCP   <- eval_metrics(predsets_GCP, y_te, K)
  classwise_GCP <- compute_classwise(predsets_GCP, y_te, K)
  
  # ----- CC-CP -----
  scores_by_label_cal <- split(s_cal_true, y_cal)
  label_clusters <- label_kmeans(scores_by_label_cal, Kc = Kc,
                                 alpha = alpha,
                                 quantiles = c(.1,.3,.5,.7,.9,1-alpha))
  label_clusters <- merge_small_clusters(label_clusters, y_cal, m_min = 100)
  Kc_eff <- length(unique(label_clusters))
  
  qC_cal <- sapply(seq_len(Kc_eff), function(cc){
    idx <- which(label_clusters[y_cal] == cc)
    conformal_quantile(s_cal_true[idx], alpha)
  })
  size_cc <- sapply(seq_len(Kc_eff), function(cc) sum(label_clusters[y_cal] == cc))
  tau <- 0.0 + 0.05 * as.numeric(size_cc < 150) + 0.03 * as.numeric(size_cc < 80)
  qC_cal_safe <- qC_cal + tau
  
  predsets_CCCP <- lapply(seq_len(nrow(p_tst)), function(i){
    pr <- p_tst[i, ]; names(pr) <- colnames(p_tst)
    build_pred_set_cluster(pr, label_clusters, qC_cal_safe)
  })
  metrics_CCCP   <- eval_metrics(predsets_CCCP, y_te, K)
  classwise_CCCP <- compute_classwise(predsets_CCCP, y_te, K)
  
  # ----- SCC-CP -----
  qC_sel <- sapply(seq_len(Kc_eff), function(cc){
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
    grid               = seq(0, 1, by = 0.1)
  )
  
  q_star <- (1 - lambda_hat) * qC_cal_safe + lambda_hat * qG_cal
  
  predsets_SCCP <- lapply(seq_len(nrow(p_tst)), function(i){
    pr <- p_tst[i, ]; names(pr) <- colnames(p_tst)
    build_pred_set_cluster(pr, label_clusters, q_star)
  })
  metrics_SCCP   <- eval_metrics(predsets_SCCP, y_te, K)
  classwise_SCCP <- compute_classwise(predsets_SCCP, y_te, K)
  
  list(
    GCP   = c(metrics_GCP,  list(classwise = classwise_GCP)),
    CCCP  = c(metrics_CCCP, list(classwise = classwise_CCCP)),
    SCCP  = c(metrics_SCCP, list(classwise = classwise_SCCP)),
    lambda_hat = lambda_hat,
    qG_cal = qG_cal, qC_cal = qC_cal_safe, q_star = q_star,
    label_clusters = label_clusters
  )
}

