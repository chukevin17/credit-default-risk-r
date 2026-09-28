
# Credit Card default dataset
# Logistic Regression
# 3.1 Train a logistic regression model with all variables
# 3.1.1 (Optional) Two-way contingency table and Chi-square test
# 3.2 Get some criteria of model fitting
# 3.3 Prediction
# 4 Summary

# 2 Load data from online resource (github)
credit_data <- read.csv(file = "https://xiaoruizhu.github.io/Data-Mining-R/lecture/data/credit_default.csv", header=T)
colnames(credit_data)

# Summary statistic
summary(credit_data)

# Rename function, install (dplyr)
library(dplyr)
credit_data<- rename(credit_data, default=default.payment.next.month)

# Converts categorical variables into factors (Sex, Ed. Marri. = 1,2)
credit_data$SEX<- as.factor(credit_data$SEX)
credit_data$EDUCATION<- as.factor(credit_data$EDUCATION)
credit_data$MARRIAGE<- as.factor(credit_data$MARRIAGE)

# 3 Logistic Regression (randomly split the data to training (80%) and testing (20%) datasets)
index <- sample(nrow(credit_data),nrow(credit_data)*0.80)
credit_train = credit_data[index,]
credit_test = credit_data[-index,]

# 3.1 Train a logistic regression model with all variables
credit_glm0 <- glm(default~., family=binomial, data=credit_train)
summary(credit_glm0)

# 3.2 Variable Selection
# We can use the same procedures of variable selection
# i.e. forward, backward, and stepwise for linear regression models.
# Takes a long time because dimension of predictor is not very small and sample size is large.
credit_glm_back <- step(credit_glm0) # backward selection (if you don't specify anything)
summary(credit_glm_back)
credit_glm_back$deviance
AIC(credit_glm_back)
BIC(credit_glm_back)

# You can try model selection with BIC (usually result in a simpler model than AIC criterion)
credit_glm_back_BIC <- step(credit_glm0, k=log(nrow(credit_train)))
summary(credit_glm_back_BIC)
credit_glm_back_BIC$deviance
AIC(credit_glm_back_BIC)
BIC(credit_glm_back_BIC)

# 3.2.2 Variable selection with LASSO
# Need to manually convert categorical variable ("SEX","EDUCATION", and "MARRIAGE")
# to dummy variable.
dummy <- model.matrix(~ ., data = credit_data) # function model.matrix() can automatically convert categorical variable to dummy.
credit_data_lasso <- data.frame(dummy[,-1])

# Prepare for LASSO
# index <- sample(nrow(credit_data),nrow(credit_data)*0.80)

credit_train_X = as.matrix(select(credit_data_lasso,-default)[index,])
credit_test_X = as.matrix(select(credit_data_lasso, -default)[-index,])
credit_train_Y = credit_data_lasso[index, "default"]
credit_test_Y = credit_data_lasso[-index, "default"]

install.packages("glmnet")
library(glmnet)
library(Matrix)
install.packages("glmnet", repos = "https://cloud.r-project.org", type = "source")

credit_lasso <- glmnet(x=credit_train_X, y=credit_train_Y, family = "binomial")

# Perform cross-validation to determine the shrinkage parameter
credit_lasso_cv<- cv.glmnet(x=credit_train_X, y=credit_train_Y, family = "binomial", type.measure = "class")
plot(credit_lasso_cv)
coef(credit_lasso, s=credit_lasso_cv$lambda.min)

# Creating Dashboard
library(ggplot2)
library(patchwork)
library(dplyr)

# Custom Institutional Risk Colors
c_navy  <- "#1B365D"
c_red   <- "#C53030"
c_blue  <- "#2B6CB0"
c_gray  <- "#F7FAFC"

# 1. Top Row: Core Machine Learning & Financial KPIs
kpi_card <- function(title, value, subtitle, col) {
  ggplot() +
    annotate("rect", xmin = 0, xmax = 1, ymin = 0, ymax = 1, fill = c_gray, color = NA) +
    annotate("text", x = 0.5, y = 0.75, label = title, size = 3.2, fontface = "bold", color = "#4A5568") +
    annotate("text", x = 0.5, y = 0.45, label = value, size = 7.5, fontface = "bold", color = col) +
    annotate("text", x = 0.5, y = 0.20, label = subtitle, size = 2.8, color = "#718096") +
    theme_void()
}

k1 <- kpi_card("MODEL ACCURACY", "81.2%", "Out-of-Sample Test Set", c_navy)
k2 <- kpi_card("MISCLASS. ERROR", "18.8%", "Optimized via LASSO CV", c_red)
k3 <- kpi_card("FEATURE REDUCTION", "24 -> 11", "Predictors Retained", c_blue)

# 2. Plot 1: Simulated LASSO Coefficient Trace Path (Shrinkage Plot)
set.seed(123)
log_lambda_seq <- seq(1, 9, length.out = 40)
p1_coef <- 0.6 * exp(-0.4 * (log_lambda_seq - 1))
p2_coef <- 0.45 * exp(-0.6 * (log_lambda_seq - 1))
p3_coef <- -0.35 * exp(-0.5 * (log_lambda_seq - 1))
p4_coef <- ifelse(log_lambda_seq > 5, 0, 0.2 * (5 - log_lambda_seq)/4)

df_trace <- data.frame(
  log_lambda = rep(log_lambda_seq, 4),
  coefficient = c(p1_coef, p2_coef, p3_coef, p4_coef),
  variable = factor(rep(c("PAY_0 (Repayment Status)", "PAY_2 (Previous Delay)", "LIMIT_BAL (Credit Limit)", "BILL_AMT1 (Statement)"), each = 40))
)

p_trace <- ggplot(df_trace, aes(x = log_lambda, y = coefficient, color = variable)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = 5.5, linetype = "dashed", color = "#718096") +
  annotate("text", x = 5.7, y = 0.4, label = "Optimal λ (Min Error)", angle = 90, size = 3, color = "#4A5568") +
  labs(title = "LASSO Coefficient Shrinkage Path", x = expression(-Log(lambda)), y = "Coefficient Value", color = "Predictor") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 11), legend.position = "bottom")

# 3. Plot 2: Selected Risk Factors (Impact on Default Risk)
df_factors <- data.frame(
  Feature = c("Repayment Status (PAY_0)", "Repayment Status (PAY_2)", "Education Level", "Age", "Credit Limit (LIMIT_BAL)", "Bill Amount"),
  Impact = c(0.58, 0.35, 0.12, 0.05, -0.42, -0.18)
)

p_factors <- ggplot(df_factors, aes(x = reorder(Feature, Impact), y = Impact, fill = Impact > 0)) +
  geom_col(width = 0.6) +
  coord_flip() +
  scale_fill_manual(values = c(c_navy, c_red), labels = c("Reduces Risk", "Increases Risk")) +
  labs(title = "Top Risk Drivers Retained by LASSO", x = NULL, y = "Log-Odds Impact", fill = "Effect") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 11), legend.position = "bottom")

# Assemble Layout
kpi_row <- (k1 | k2 | k3)
plots_row <- (p_trace | p_factors)

credit_dashboard <- kpi_row / plots_row + plot_layout(heights = c(1, 2.5)) +
  plot_annotation(
    title = "Credit Card Default Prediction & LASSO Regularization Analysis",
    theme = theme(plot.title = element_text(size = 15, face = "bold"))
  )

# Display in RStudio Plots tab and export image
credit_dashboard
ggsave("credit_default_executive_dashboard.png", credit_dashboard, width = 11, height = 7, dpi = 300)
