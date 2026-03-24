# ============================================================
# Reconstruction performance (R2, MSE) + Classification performance (AUC, Accuracy)
# Top row   : R2, MSE
# Bottom row: AUC, Accuracy
# ============================================================

library(tidyverse)
library(patchwork)
library(scales)
library(grid)

# ------------------------------------------------------------
# 1. EXACT RESULTS TABLES
# ------------------------------------------------------------

r2_df <- tribble(
  ~Disease, ~Winning_config, ~Best_BetaVAE_model, ~BaselineVAE, ~Best_BetaVAE, ~gVAE,
  "ALZ", "2000_30_3", "BetaVAE_NS1_beta4.0",  0.6106922626495361, 0.609825849533081,  0.6247621774673462,
  "ASD", "500_150_3", "BetaVAE_NS1_beta2.0",  0.2260158061981201, 0.2262905836105346, 0.2323955297470092,
  "BD",  "300_50_3",  "BetaVAE_NS1_beta1.0",  0.5089702606201172, 0.5056077241897583, 0.5188989639282227,
  "BMI", "1000_200_3","BetaVAE_NS1_beta1.0",  0.2264045476913452, 0.2240012884140014, 0.2313598394393921,
  "BRC", "100_150_3", "BetaVAE_NS1_beta10.0", 0.2272084951400756, 0.2254742383956909, 0.2296843528747558,
  "CAD", "1000_200_3","BetaVAE_NS1_beta4.0",  0.5026177167892456, 0.5021295547485352, 0.5204927921295166,
  "CD",  "500_10_4",  "BetaVAE_NS1_beta10.0", 0.5108280181884766, 0.5054274797439575, 0.5174707174301147,
  "COL", "2000_50_3", "BetaVAE_NS1_beta2.0",  0.2234412431716919, 0.2213938236236572, 0.2283813953399658,
  "EOS", "500_100_3", "BetaVAE_NS1_beta4.0",  0.2239528894424438, 0.2244377136230468, 0.2302440404891967,
  "HDL", "1500_100_3","BetaVAE_NS1_beta4.0",  0.2254906892776489, 0.2230046987533569, 0.2288961410522461,
  "HGT", "500_200_3", "BetaVAE_NS1_beta10.0", 0.2255029678344726, 0.224324345588684,  0.2314707040786743,
  "HT",  "300_200_3", "BetaVAE_NS1_beta2.0",  0.5083924531936646, 0.504989743232727,  0.5178546905517578,
  "LDL", "300_10_4",  "BetaVAE_NS1_beta10.0", 0.2278167009353637, 0.2257157564163208, 0.2317585945129394,
  "LUN", "1500_100_3","BetaVAE_NS1_beta2.0",  0.2242103815078735, 0.2219314575195312, 0.2284609079360962,
  "PRC", "500_50_3",  "BetaVAE_NS1_beta1.0",  0.2232238054275512, 0.2228304147720337, 0.2295091152191162,
  "RA",  "1000_100_3","BetaVAE_NS1_beta1.0",  0.5021249055862427, 0.5044149160385132, 0.517654538154602,
  "T1D", "1500_150_3","BetaVAE_NS1_beta10.0", 0.4978233575820923, 0.4997483491897583, 0.5123106241226196,
  "T2D", "1500_200_3","BetaVAE_NS1_beta4.0",  0.505638599395752,  0.501976490020752,  0.5167604684829712
)

mse_df <- tribble(
  ~Disease, ~Winning_config, ~Best_BetaVAE_model, ~BaselineVAE, ~Best_BetaVAE, ~gVAE,
  "ALZ", "50_30_2",   "BetaVAE_NS1_beta4.0",  0.7014084,  0.7008923,  0.6984466,
  "ASD", "150_10_2",  "BetaVAE_NS1_beta4.0",  0.3125184,  0.31201315, 0.3112303,
  "BD",  "1500_10_3", "BetaVAE_NS1_beta1.0",  0.62310463, 0.62111896, 0.61565995,
  "BMI", "100_10_2",  "BetaVAE_NS1_beta1.0",  0.3099218,  0.30962297, 0.30784634,
  "BRC", "200_30_2",  "BetaVAE_NS1_beta2.0",  0.31116292, 0.3097628,  0.3044817,
  "CAD", "1500_200_2","BetaVAE_NS1_beta10.0", 0.6164986,  0.61638945, 0.6112788,
  "CD",  "50_10_2",   "BetaVAE_NS1_beta4.0",  0.6219995,  0.6222573,  0.6192775,
  "COL", "1500_200_2","BetaVAE_NS1_beta2.0",  0.31135643, 0.31135184, 0.3085121,
  "EOS", "10_150_2",  "BetaVAE_NS1_beta1.0",  0.31655934, 0.31636736, 0.3143928,
  "HDL", "50_10_2",   "BetaVAE_NS1_beta10.0", 0.31344342, 0.31286812, 0.31118155,
  "HGT", "200_100_2", "BetaVAE_NS1_beta1.0",  0.30821535, 0.30829054, 0.3056116,
  "HT",  "2000_10_2", "BetaVAE_NS1_beta1.0",  0.6174503,  0.6158403,  0.60882175,
  "LDL", "200_100_2", "BetaVAE_NS1_beta4.0",  0.3080554,  0.30686557, 0.30397,
  "LUN", "2000_30_2", "BetaVAE_NS1_beta10.0", 0.3112213,  0.31089926, 0.30796993,
  "PRC", "1500_30_2", "BetaVAE_NS1_beta10.0", 0.3113779,  0.31125924, 0.3085889,
  "RA",  "2000_200_2","BetaVAE_NS1_beta2.0",  0.6157896,  0.61553437, 0.6108086,
  "T1D", "2000_150_3","BetaVAE_NS1_beta10.0", 0.61532044, 0.61667395, 0.61048436,
  "T2D", "200_10_2",  "BetaVAE_NS1_beta4.0",  0.61818,    0.6184932,  0.6123673
)

auc_df <- tribble(
  ~Disease, ~BaselineVAE, ~Best_BetaVAE, ~gVAE,
  "CAD",    0.52,         0.52,          0.58,
  "RA",     0.54,         0.63,          0.66,
  "COL",    0.50,         0.50,          0.53,
  "CD",     0.50,         0.50,          0.53,
  "T2D",    0.50,         0.50,          0.53,
  "BD",     0.53,         0.53,          0.55,
  "HT",     0.58,         0.58,          0.59,
  "PRC",    0.58,         0.58,          0.59,
  "LUN",    0.00,         0.50,          0.54,
  "ALZ",    0.54,         0.55,          0.58,
  "ASD",    0.52,         0.53,          0.56,
  "BRC",    0.53,         0.54,          0.57,
  "T1D",    0.54,         0.55,          0.58
)

acc_df <- tribble(
  ~Disease, ~BaselineVAE, ~Best_BetaVAE, ~gVAE,
  "T2D",    0.60,         0.00,          1.00,
  "RA",     0.61,         0.62,          0.65,
  "CAD",    0.62,         0.62,          0.63,
  "CD",     0.63,         0.63,          0.64,
  "BD",     0.65,         0.66,          0.66,
  "COL",    0.61,         0.00,          0.62,
  "LUN",    0.63,         0.00,          0.63,
  "HT",     0.60,         0.60,          0.60,
  "PRC",    0.60,         0.60,          0.60,
  "ALZ",    0.63,         0.64,          0.67,
  "ASD",    0.61,         0.62,          0.65,
  "BRC",    0.64,         0.65,          0.68,
  "T1D",    0.64,         0.65,          0.68
)

to_long <- function(df, metric_name) {
  df %>%
    select(Disease, BaselineVAE, Best_BetaVAE, gVAE) %>%
    pivot_longer(cols = -Disease, names_to = "Model", values_to = "Value") %>%
    mutate(
      Metric = metric_name,
      Model = factor(
        Model,
        levels = c("BaselineVAE", "Best_BetaVAE", "gVAE"),
        labels = c("BaselineVAE", "Best BetaVAE", "gVAE")
      )
    )
}

r2_long <- to_long(r2_df, "R2")
mse_long <- to_long(mse_df, "MSE")
auc_long <- to_long(auc_df, "AUC")
acc_long <- to_long(acc_df, "Accuracy")

r2_order <- c("COL", "LUN", "HDL", "PRC", "BRC", "EOS", "BMI", "HGT", "LDL", "ASD", "T1D", "T2D", "CD", "RA", "HT", "BD", "CAD", "ALZ")
mse_order <- c("ALZ", "CD", "BD", "T2D", "CAD", "RA", "T1D", "HT", "EOS", "ASD", "HDL", "PRC", "COL", "LUN", "BMI", "HGT", "BRC", "LDL")
auc_order <- c("CAD", "RA", "COL", "CD", "T2D", "BD", "HT", "PRC", "LUN", "EOS", "ALZ", "ASD", "BMI", "BRC", "HDL", "HGT", "LDL", "T1D")
acc_order <- c("T2D", "RA", "CAD", "CD", "BD", "COL", "LUN", "HT", "PRC", "EOS", "ALZ", "ASD", "BMI", "BRC", "HDL", "HGT", "LDL", "T1D")

r2_long <- r2_long %>% mutate(Disease = factor(Disease, levels = rev(r2_order)))
mse_long <- mse_long %>% mutate(Disease = factor(Disease, levels = rev(mse_order)))
auc_long <- auc_long %>% mutate(Disease = factor(Disease, levels = rev(auc_order)))
acc_long <- acc_long %>% mutate(Disease = factor(Disease, levels = rev(acc_order)))

model_colors <- c("BaselineVAE" = "#4C78A8", "Best BetaVAE" = "#F58518", "gVAE" = "#54A24B")

theme_journal <- function() {
  theme_minimal(base_size = 13, base_family = "sans") +
    theme(
      plot.title = element_text(size = 15, face = "bold", hjust = 0),
      axis.title.x = element_text(size = 12.5, face = "bold"),
      axis.title.y = element_text(size = 12.5, face = "bold"),
      axis.text = element_text(size = 10.5, colour = "black"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(colour = "grey88", linewidth = 0.4),
      axis.line = element_line(colour = "black", linewidth = 0.5),
      legend.title = element_blank(),
      legend.text = element_text(size = 11),
      legend.position = "right",
      legend.key.height = unit(0.55, "cm"),
      legend.key.width = unit(0.9, "cm"),
      plot.margin = margin(8, 10, 8, 8),
      strip.text = element_text(face = "bold")
    )
}

make_panel <- function(data, title, xlab, xlim_max, xticks = waiver(), ylab = "Disease") {
  ggplot(data, aes(x = Value, y = Disease, fill = Model)) +
    geom_col(position = position_dodge2(width = 0.72, preserve = "single"), width = 0.68, colour = "white", linewidth = 0.25) +
    scale_fill_manual(values = model_colors, drop = FALSE) +
    scale_x_continuous(limits = c(0, xlim_max), breaks = xticks, expand = expansion(mult = c(0, 0.02))) +
    labs(title = title, x = xlab, y = ylab) +
    theme_journal()
}

p_r2 <- make_panel(r2_long, "a  R2 across diseases", expression(R^2 ~ "(higher is better)"), 0.66, seq(0, 0.6, by = 0.1), "Disease")
p_mse <- make_panel(mse_long, "b  MSE across diseases", "MSE (lower is better)", 0.75, seq(0, 0.7, by = 0.1), NULL) + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
p_auc <- make_panel(auc_long, "c  AUC across diseases", "AUC (higher is better)", 0.90, seq(0, 0.9, by = 0.1), "Disease")
p_acc <- make_panel(acc_long, "d  Accuracy across diseases", "Accuracy (higher is better)", 1.05, seq(0, 1.0, by = 0.1), NULL) + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

final_plot <- (p_r2 | p_mse) / (p_auc | p_acc) + plot_layout(guides = "collect") & theme(legend.position = "right", plot.background = element_rect(fill = "white", colour = NA))
print(final_plot)

save_pdf_device <- if (capabilities("cairo")) cairo_pdf else "pdf"
ggsave("Figure_R2_MSE_AUC_Accuracy_across_diseases_high_impact.pdf", final_plot, width = 16, height = 11, units = "in", dpi = 600, device = save_pdf_device, bg = "white")
ggsave("Figure_R2_MSE_AUC_Accuracy_across_diseases_high_impact.png", final_plot, width = 16, height = 11, units = "in", dpi = 600, bg = "white")
ggsave("Figure_R2_MSE_AUC_Accuracy_across_diseases_high_impact.tiff", final_plot, width = 16, height = 11, units = "in", dpi = 600, compression = "lzw", bg = "white")
