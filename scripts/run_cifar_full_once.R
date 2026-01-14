source("scripts/cifar_from_npz_full.R")

npz_path <- "out/cifar100_probs/cifar100_probs_seed1_e1_tr30000_sel10000_cal10000.npz"

res <- run_cifar_from_npz(npz_path, alpha = 0.1, Kc = 10, tail_frac = 0.2)

print(res$overall)
cat("\nmean lambda:", mean(res$lambda_hat, na.rm = TRUE), "\n")
cat("lambda table:\n")
print(table(round(res$lambda_hat, 2)))

print(res$tail)
cat("\n Tail table:\n")
print(res$classwise_summary)