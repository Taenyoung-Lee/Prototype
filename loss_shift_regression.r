################################################################################
#  loss_shifting_reg_v1.R   (2025-09-01)
#  - Classification code (loss_shifting_v4.R)をRegressionに適合
#  - Margin-based shift -> Residual-based shift
#  - MM/IRLS updater -> Simple Ridge Regression closed-form update
#  - Accuracy metric -> RMSE, MAE, R2 metrics
################################################################################

suppressPackageStartupMessages({ library(quadprog) }) # quadprog is not used, but kept for consistency

#################### (0) Logging Utils (Unchanged) ##############################
.ts_now <- function() format(Sys.time(), "%H:%M:%S")
.v_on  <- function(verbose) isTRUE(verbose)
.vopen <- function(tag, msg = "", verbose = FALSE) { if (.v_on(verbose)) cat(sprintf("\n┏[%s %s] %s\n", tag, .ts_now(), msg)) }
.vstep <- function(tag, msg, verbose = FALSE)      { if (.v_on(verbose)) cat(sprintf("┃ %s\n", msg)) }
.vok   <- function(tag, msg = "done", verbose = FALSE) { if (.v_on(verbose)) cat(sprintf("┗[%s %s] %s\n", tag, .ts_now(), msg)) }
.vkv <- function(k, v) sprintf("%s=%s", k, v)

#################### (1) RBF 커널 (Unchanged) ####################################
rbf_kernel <- function(X1, X2 = NULL, sigma = 1) {
  if (is.null(X2)) X2 <- X1
  dist2 <- outer(rowSums(X1^2), rowSums(X2^2), "+") - 2 * tcrossprod(X1, X2)
  exp(-dist2 / (2 * sigma^2))
}

#################### (2) [REG] Shift 함수 (Regression Version) #####################
shift_r_reg_none <- function(e) rep(0, length(e))
shift_r_reg_soft <- function(e, alpha, eta) {
  shift_val <- alpha * (abs(e) - eta)
  ifelse(abs(e) <= eta, 0, sign(e) * shift_val)
}
shift_r_reg_hard <- function(e, alpha, eta) {
  shift_val <- alpha * abs(e)
  ifelse(abs(e) <= eta, 0, sign(e) * shift_val)
}

#################### (3) [REG] 목적 함수 (Regression Version) ####################
objective_value_reg <- function(theta, X, y, shift_fun, lambda) {
  X_aug <- cbind(1, X)
  f_val <- as.vector(X_aug %*% theta)
  e_vec <- y - f_val
  r_vec <- shift_fun(e_vec)
  eff_e <- e_vec - r_vec
  # [REG] MSE + L2 penalty
  mean(eff_e^2) + lambda * sum(theta[-1]^2) / 2
}
make_penalty_mat <- function(p, n, lambda) diag(c(0, rep(n * lambda, p)))

#################### (4) Z-score Helper (Unchanged) ###########################
.standardize_fit <- function(X) {
  mu <- colMeans(X)
  sd <- apply(X, 2, sd); sd[sd == 0] <- 1
  Xs <- sweep(sweep(X, 2, mu, "-"), 2, sd, "/")
  list(Xs = Xs, mu = mu, sd = sd)
}
.standardize_apply <- function(X, mu, sd) {
  sd[sd == 0] <- 1
  sweep(sweep(X, 2, mu, "-"), 2, sd, "/")
}

#################### (5) [REG] 주 학습 함수 (Regression Version) ################
loss_shift_regressor_all <- function(
    X, y,
    style     = c("soft", "hard", "none"),
    kernel    = c("linear", "gaussian"),
    sigma = 1, alpha = 1, eta = 0.5,
    lambda = 0.1, max_iter = 100, tol = 1e-7,
    init_theta = NULL,
    verbose = FALSE,
    standardize = TRUE
) {
  tag <- "TRAIN(REG)"
  X_mat <- as.matrix(X)
  y_vec <- as.numeric(y)
  style     <- match.arg(style)
  kernel    <- match.arg(kernel)
  
  .vopen(tag, sprintf("START  [%s | %s]", style, kernel), verbose)
  
  if (standardize) {
    st <- .standardize_fit(X_mat)
    X_proc <- st$Xs
  } else {
    st <- NULL; X_proc <- X_mat
  }
  
  if (kernel == "gaussian") {
    K_train <- rbf_kernel(X_proc, NULL, sigma)
    eig     <- eigen(K_train, symmetric = TRUE)
    vals    <- pmax(eig$values, .Machine$double.eps)
    U       <- eig$vectors
    D_half  <- diag(sqrt(vals))
    X_feat  <- U %*% D_half
  } else X_feat <- X_proc
  
  shift_fun <- switch(style,
                      soft = function(e) shift_r_reg_soft(e, alpha, eta),
                      hard = function(e) shift_r_reg_hard(e, alpha, eta),
                      none = function(e) shift_r_reg_none(e))

  n <- nrow(X_feat); p <- ncol(X_feat)
  theta <- if (!is.null(init_theta) && length(init_theta) == p + 1) init_theta else rep(0, p + 1)
  X_aug <- cbind(1, X_feat)
  
  obj_old <- objective_value_reg(theta, X_feat, y_vec, shift_fun, lambda)
  .vstep(tag, paste(
    .vkv("n", n), .vkv("p", p),
    .vkv("lambda", format(lambda, digits=4)),
    .vkv("sigma", if (kernel=="gaussian") format(sigma, digits=4) else "NA"),
    .vkv("alpha", if (style!="none") format(alpha, digits=4) else "NA"),
    .vkv("eta",   if (style!="none") format(eta,   digits=4) else "NA"),
    sep = " | "
  ), verbose)
  
  iter_used <- NA
  for (it in seq_len(max_iter)) {
    # [REG] Step 1: Calculate residuals and shifts
    f_val <- as.vector(X_aug %*% theta)
    e_vec <- y_vec - f_val
    r_vec <- shift_fun(e_vec)
    y_eff <- y_vec - r_vec
    
    # [REG] Step 2: Update theta with a simple Ridge Regression solution
    # This replaces the complex MM/IRLS/QP updaters
    A <- crossprod(X_aug) + diag(c(0, rep(lambda, p))) # Simplified penalty
    b <- crossprod(X_aug, y_eff)
    theta_new <- solve(A, b)
    
    obj_new <- objective_value_reg(theta_new, X_feat, y_vec, shift_fun, lambda)
    rel <- abs(obj_new - obj_old) / (abs(obj_old) + 1e-8)
    .vstep(tag, sprintf("[iter=%03d] obj=%.6g  rel=%.3e", it, obj_new, rel), verbose)
    
    theta <- theta_new; obj_old <- obj_new
    if (rel < tol) { iter_used <- it; break }
    if (it == max_iter) iter_used <- it
  }
  
  .vok(tag, sprintf("END (iters=%d, final_obj=%.6g)", iter_used, obj_old), verbose)
  
  out <- list(theta = theta, style = style, kernel = kernel,
              sigma = sigma, alpha = alpha, eta = eta, lambda = lambda,
              iter_used = iter_used,
              standardize = standardize)
  
  if (isTRUE(standardize)) { out$center <- st$mu; out$scale <- st$sd }
  if (kernel == "gaussian") {
    out$U <- U; out$D_inv_sqrt <- 1 / sqrt(vals); out$X_train_orig <- X_mat
  }
  out
}

#################### (6) [REG] 예측 함수 #######################################
predict_loss_shift_reg <- function(model, X_new) {
  X_new_mat <- as.matrix(X_new)
  if (isTRUE(model$standardize)) {
    X_proc <- .standardize_apply(X_new_mat, model$center, model$scale)
  } else X_proc <- X_new_mat
  
  if (model$kernel == "gaussian") {
    X_tr <- model$X_train_orig
    X_tr_proc <- if (isTRUE(model$standardize)) .standardize_apply(X_tr, model$center, model$scale) else X_tr
    K_new <- rbf_kernel(X_proc, X_tr_proc, model$sigma)
    Z_new <- K_new %*% model$U %*% diag(model$D_inv_sqrt)
    X_aug <- cbind(1, Z_new)
  } else X_aug <- cbind(1, X_proc)
  
  as.vector(X_aug %*% model$theta)
}

#################### (7) [REG] 평가 지표 #######################################
evaluate_metrics_reg <- function(truth, pred) {
  residuals <- truth - pred
  mae <- mean(abs(residuals))
  rmse <- sqrt(mean(residuals^2))
  r2 <- 1 - sum(residuals^2) / sum((truth - mean(truth))^2)
  list(mae = mae, rmse = rmse, r.squared = r2)
}

#################### (8) CV 분할 (Unchanged) ####################################
cv_split <- function(n, n_folds, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  folds <- sample(rep(seq_len(n_folds), length.out = n))
  lapply(seq_len(n_folds), function(k) list(tr = which(folds != k), va = which(folds == k)))
}

#################### (9) [REG] CV + restart 함수 ##############################
cv_metric_restart_reg <- function(
    X, y, folds,
    style, kernel,
    lambda, sigma, alpha, eta,
    restarts, max_iter,
    base_seed = NULL,
    verbose = FALSE) {
  
  tag <- "CV-RMSE(RESTART)"
  best_rmse <- Inf
  .vopen(tag, sprintf("START %s", paste(
    .vkv("lambda", format(lambda, digits=4)),
    .vkv("sigma",  format(sigma,  digits=4)),
    .vkv("alpha",  if (style!="none") format(alpha, digits=4) else "NA"),
    .vkv("eta",    if (style!="none") format(eta,   digits=4) else "NA"),
    sep = " | ")), verbose)
  
  for (r in seq_len(restarts)) {
    rmses <- numeric(length(folds))
    for (i in seq_along(folds)) {
      tr <- folds[[i]]$tr; va <- folds[[i]]$va
      p_tr <- if (kernel == "gaussian") length(tr) else ncol(X)
      if (!is.null(base_seed)) set.seed(base_seed + r * 10000 + i)
      init_th <- rnorm(p_tr + 1, 0, 0.1)
      
      mod <- loss_shift_regressor_all(
        X[tr, ], y[tr],
        style = style, kernel = kernel,
        sigma = sigma, alpha = alpha, eta = eta, lambda = lambda,
        max_iter = max_iter, init_theta = init_th, 
        verbose = FALSE, standardize = TRUE)
      
      preds <- predict_loss_shift_reg(mod, X[va, ])
      rmses[i] <- sqrt(mean((y[va] - preds)^2))
    }
    cur <- mean(rmses, na.rm = TRUE)
    best_rmse <- min(best_rmse, cur)
    .vstep(tag, sprintf("restart=%02d  fold_mean_rmse=%.4f  best=%.4f", r, cur, best_rmse), verbose)
  }
  .vok(tag, sprintf("END   best_cv_rmse=%.4f", best_rmse), verbose)
  best_rmse
}

#################### (10) [REG] 풀 그리드 튜너 #################################
fullgrid_tune_loss_shift_reg <- function(
    X, y,
    style, kernel,
    lambda_grid, alpha_grid, eta_grid,
    restarts = 5, n_folds = 5,
    max_iter = 100,
    verbose = FALSE,
    seed = NULL,
    sigma_mult = c(0.5, 1, 2)) {
  
  tag <- "TUNE-GRID(REG)"
  .vopen(tag, sprintf("START [%s | %s] grid-search", style, kernel), verbose)
  
  sigma_grid_eff <- if (kernel == "gaussian") {
     if (!is.null(seed)) set.seed(seed + 777)
     Xs <- scale(as.matrix(X))
     m  <- min(nrow(Xs), 400)
     idx <- if (nrow(Xs) > m) sample(nrow(Xs), m) else seq_len(nrow(Xs))
     D  <- as.matrix(dist(Xs[idx, , drop = FALSE]))
     med2 <- median(D[upper.tri(D)]^2)
     sigma_med <- sqrt(pmax(med2, .Machine$double.eps) / 2)
     sort(unique(sigma_med * sigma_mult))
  } else {
     1
  }

  grid <- expand.grid(lambda = lambda_grid,
                      sigma  = sigma_grid_eff,
                      alpha  = if (style == "none") 1 else alpha_grid,
                      eta    = if (style == "none") 0 else eta_grid)
  
  folds <- cv_split(nrow(X), n_folds, seed = seed)
  
  metric_vec <- numeric(nrow(grid))
  for (i in seq_len(nrow(grid))) {
    metric_vec[i] <- cv_metric_restart_reg(
      X, y, folds,
      style, kernel,
      lambda = grid$lambda[i], sigma  = grid$sigma[i],
      alpha  = grid$alpha[i], eta    = grid$eta[i],
      restarts = restarts, max_iter = max_iter,
      base_seed = if (is.null(seed)) NULL else seed + i * 1000,
      verbose = verbose)
    
    cat(sprintf("(lambda=%.3g, sigma=%.3g, alpha=%.3g, eta=%.3g) -> cv-rmse=%.4f [%d/%d]\n",
                grid$lambda[i], grid$sigma[i], grid$alpha[i], grid$eta[i], metric_vec[i], i, nrow(grid)))
  }
  
  # [REG] Find best parameters by MINIMIZING RMSE
  best_metric <- min(metric_vec, na.rm = TRUE)
  tie_idx  <- which(metric_vec == best_metric)
  
  if (length(tie_idx) == 1) {
    best_idx <- tie_idx
  } else {
    G <- grid[tie_idx, , drop = FALSE]
    o <- order( -G$lambda, if (kernel == "gaussian") -G$sigma else 0,
                if (style != "none") G$alpha else 0, if (style != "none") -G$eta else 0)
    best_idx <- tie_idx[o[1]]
  }
  
  .vok(tag, sprintf("END best at #%d (cv_rmse=%.4f)", best_idx, metric_vec[best_idx]), verbose)
  list(best_pars = grid[best_idx, , drop = FALSE],
       best_cv_metric = metric_vec[best_idx],
       cv_table = cbind(grid, cv_rmse = metric_vec))
}


#################### (11) [REG] 통합 Wrapper #################################
train_loss_shift_reg <- function(
    X_train, y_train,
    X_test = NULL,
    style     = c("soft", "hard", "none"),
    kernel    = c("linear", "gaussian"),
    lambda_grid = 2^seq(-10, 5, len = 10),
    alpha_grid  = c(0.25, 0.5, 0.75, 1),
    eta_grid    = c(0.5, 1, 1.5, 2),
    restarts    = 5,
    n_folds     = 5,
    max_iter    = 100,
    verbose     = FALSE,
    seed        = NULL
) {
  tag <- "TRAIN(REG-WRAP)"
  style     <- match.arg(style)
  kernel    <- match.arg(kernel)
  if (!is.null(seed)) set.seed(seed)
  
  .vopen(tag, sprintf("START [%s | %s]", style, kernel), verbose)
  
  # When eta needs to be data-driven, calculate it here
  # For example, eta_grid can be defined as quantiles of |y - mean(y)|
  y_sd <- sd(y_train)
  eta_grid_scaled <- eta_grid * y_sd
  cat("Scaled eta_grid based on sd(y):", round(eta_grid_scaled, 2), "\n")
  
  tune_out <- fullgrid_tune_loss_shift_reg(
    X_train, y_train,
    style = style, kernel = kernel,
    lambda_grid = lambda_grid, alpha_grid = alpha_grid, eta_grid = eta_grid_scaled,
    restarts = restarts, n_folds = n_folds, max_iter = max_iter,
    verbose = verbose,
    seed = if (is.null(seed)) NULL else seed + 1234)
  
  best <- tune_out$best_pars
  cat("\n[train_loss_shift_reg] Best hyper-parameters:\n"); print(best)
  cat(sprintf("Best CV RMSE = %.4f\n\n", tune_out$best_cv_metric))

  p_full <- if (kernel == "gaussian") nrow(X_train) else ncol(X_train)
  best_model <- NULL; best_obj <- Inf
  for (r in seq_len(restarts)) {
    if (!is.null(seed)) set.seed(seed + 200000 + r)
    init_th <- rnorm(p_full + 1, 0, 0.1)
    
    m <- loss_shift_regressor_all(
      X_train, y_train,
      style = style, kernel = kernel,
      sigma = best$sigma, alpha = best$alpha, eta = best$eta, lambda = best$lambda,
      max_iter = max_iter, init_theta = init_th, 
      verbose = verbose, standardize = TRUE)
      
    obj <- objective_value_reg(m$theta, m$X_feat, y_train, 
                               switch(style, soft=function(e)shift_r_reg_soft(e,best$alpha,best$eta),
                                      hard=function(e)shift_r_reg_hard(e,best$alpha,best$eta),
                                      none=shift_r_reg_none), m$lambda)
                                      
    if (.v_on(verbose)) .vstep(tag, sprintf("Final restart=%02d  obj=%.6g", r, obj), TRUE)
    if (obj < best_obj) { best_obj <- obj; best_model <- m }
  }
  
  test_pred <- if (!is.null(X_test)) predict_loss_shift_reg(best_model, X_test) else NULL
  .vok(tag, "END training wrapper", verbose)
  
  list(model = best_model, best_param = best,
       cv_table = tune_out$cv_table, cv_best_rmse = tune_out$best_cv_metric,
       test_pred = test_pred)
}

#################### (12) 사용 예제 ########################################
# if (sys.nframe() == 0) { # Run only when script is executed directly
  
#   cat("\n\n--- Running Loss Shifting Regressor Example ---\n")
  
#   # 1. Generate synthetic data with outliers
#   set.seed(42)
#   n <- 200
#   p <- 5
#   X <- matrix(rnorm(n * p), n, p)
#   beta_true <- c(2, -1.5, 1, 0, 0.5)
#   y_true <- X %*% beta_true + rnorm(n, 0, 0.5)
#   y <- y_true
  
#   # Add some outliers
#   outlier_idx <- sample(n, 15)
#   y[outlier_idx] <- y[outlier_idx] + rnorm(15, 0, 10)
  
#   # Split data
#   train_idx <- sample(n, floor(0.75 * n))
#   X_train <- X[train_idx, ]
#   y_train <- y[train_idx]
#   X_test <- X[-train_idx, ]
#   y_test <- y[-train_idx]
  
#   # 2. Train the robust regressor (Loss Shifting)
#   cat("\n--- Training Robust Model (style='soft') ---\n")
#   robust_model_fit <- train_loss_shift_reg(
#     X_train, y_train, X_test,
#     style = "soft",
#     kernel = "linear",
#     restarts = 2, # Keep low for a quick example
#     n_folds = 3,  # Keep low for a quick example
#     seed = 123
#   )
  
#   # 3. For comparison, train a standard Ridge regressor
#   cat("\n\n--- Training Standard Ridge Model (style='none') ---\n")
#   ridge_model_fit <- train_loss_shift_reg(
#     X_train, y_train, X_test,
#     style = "none",
#     kernel = "linear",
#     restarts = 1,
#     n_folds = 3,
#     seed = 123
#   )
  
#   # 4. Evaluate and compare performance
#   cat("\n\n--- Performance Comparison on Test Set ---\n")
  
#   metrics_robust <- evaluate_metrics_reg(y_test, robust_model_fit$test_pred)
#   metrics_ridge <- evaluate_metrics_reg(y_test, ridge_model_fit$test_pred)
  
#   cat("Robust Model (Loss Shifting) Metrics:\n")
#   print(metrics_robust)
  
#   cat("\nStandard Ridge Model Metrics:\n")
#   print(metrics_ridge)
  
#   cat("\nCONCLUSION: The robust model should have a lower MAE/RMSE,\n")
#   cat("indicating it was less affected by the outliers.\n")
  
#   # Plot predictions vs. true values
#   par(mfrow=c(1, 2), mar=c(4, 4, 3, 1))
#   plot(y_test, robust_model_fit$test_pred, main="Robust Model Predictions", 
#        xlab="True Y", ylab="Predicted Y", pch=16, col='blue')
#   abline(0, 1, col='red', lwd=2)
  
#   plot(y_test, ridge_model_fit$test_pred, main="Standard Ridge Predictions", 
#        xlab="True Y", ylab="Predicted Y", pch=16, col='darkgreen')
#   abline(0, 1, col='red', lwd=2)
  
# }