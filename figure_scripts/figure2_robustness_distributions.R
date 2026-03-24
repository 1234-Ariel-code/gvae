# ============================================================
# High-impact journal figure:
# Robustness distributions across configurations
# comparing BaselineVAE, Best BetaVAE, and gVAE
# Revised version: clearer separation between disease blocks
# ============================================================

library(tidyverse)
library(patchwork)
library(scales)
library(grid)

# ------------------------------------------------------------
# 1. READ DATA
# ------------------------------------------------------------
df <- read_csv("~/Documents/qgvae_vs_others_Robustness.csv", show_col_types = FALSE)

df <- df %>%
  filter(metric == "Robustness_invNoiseSens")

# ------------------------------------------------------------
# 2. RESHAPE TO LONG FORMAT
# ------------------------------------------------------------
plot_df <- df %>%
  select(disease, config, qg_value, baseline_value, best_beta_value) %>%
  pivot_longer(
    cols = c(qg_value, baseline_value, best_beta_value),
    names_to = "Model",
    values_to = "Robustness"
  ) %>%
  mutate(
    Model = recode(
      Model,
      "baseline_value" = "BaselineVAE",
      "best_beta_value" = "BetaVAE",
      "qg_value" = "gVAE"
    )
  )

# ------------------------------------------------------------
# 3. ORDER DISEASES BY gVAE MEDIAN
# ------------------------------------------------------------
disease_order <- plot_df %>%
  filter(Model == "gVAE") %>%
  group_by(disease) %>%
  summarise(gvae_median = median(Robustness, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(gvae_median)) %>%
  pull(disease)

plot_df <- plot_df %>%
  mutate(
    disease = factor(disease, levels = disease_order),
    Model = factor(Model, levels = c("BaselineVAE", "BetaVAE", "gVAE"))
  )

# ------------------------------------------------------------
# 4. SUMMARY TABLES
# ------------------------------------------------------------
summary_df <- plot_df %>%
  group_by(disease, Model) %>%
  summarise(
    median_robustness = median(Robustness, na.rm = TRUE),
    mean_robustness = mean(Robustness, na.rm = TRUE),
    q1 = quantile(Robustness, 0.25, na.rm = TRUE),
    q3 = quantile(Robustness, 0.75, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    disease = factor(disease, levels = disease_order)
  )

n_cfg_df <- df %>%
  group_by(disease) %>%
  summarise(n_configs = n_distinct(config), .groups = "drop") %>%
  mutate(
    disease = factor(disease, levels = disease_order)
  )

# ------------------------------------------------------------
# 5. COLORS
# ------------------------------------------------------------
model_colors <- c(
  "BaselineVAE" = "#4C78A8",
  "BetaVAE" = "#F58518",
  "gVAE" = "#54A24B"
)

# ------------------------------------------------------------
# 6. THEME
# ------------------------------------------------------------
theme_nature <- function() {
  theme_minimal(base_size = 13, base_family = "sans") +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_line(colour = "grey88", linewidth = 0.4),
      axis.line = element_line(colour = "black", linewidth = 0.45),
      axis.text = element_text(colour = "black", size = 11.5),
      axis.title = element_text(colour = "black", size = 13, face = "bold"),
      plot.title = element_text(size = 16, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 11, colour = "grey30", hjust = 0),
      legend.title = element_blank(),
      legend.text = element_text(size = 11),
      legend.position = "top",
      legend.justification = "left",
      plot.margin = margin(10, 12, 10, 10)
    )
}

# ------------------------------------------------------------
# 7. MAIN PANEL
# ------------------------------------------------------------
main_dodge <- 0.72

p_main <- ggplot(plot_df, aes(x = Robustness, y = disease, fill = Model, colour = Model)) +
  geom_violin(
    position = position_dodge(width = main_dodge),
    alpha = 0.22,
    linewidth = 0.35,
    trim = FALSE,
    width = 0.82,
    scale = "width"
  ) +
  geom_boxplot(
    position = position_dodge(width = main_dodge),
    width = 0.16,
    outlier.shape = NA,
    linewidth = 0.38,
    alpha = 0.95
  ) +
  geom_point(
    data = summary_df,
    aes(x = median_robustness, y = disease, fill = Model),
    shape = 21,
    size = 2.5,
    stroke = 0.3,
    colour = "white",
    position = position_dodge(width = main_dodge),
    inherit.aes = FALSE
  ) +
  scale_fill_manual(values = model_colors) +
  scale_colour_manual(values = model_colors) +
  scale_y_discrete(expand = expansion(add = 0.55)) +
  labs(
    title = "a  Robustness distributions across configurations",
    subtitle = "Each disease summarizes all evaluated configurations for BaselineVAE, BetaVAE, and gVAE",
    x = "Robustness (higher is better)",
    y = NULL
  ) +
  theme_nature() +
  theme(legend.position = "top")

# ------------------------------------------------------------
# 8. COMPANION PANEL
# ------------------------------------------------------------
p_summary <- ggplot(summary_df, aes(x = median_robustness, y = disease, colour = Model)) +
  geom_point(size = 2.8, position = position_dodge(width = 0.42)) +
  scale_colour_manual(values = model_colors) +
  scale_y_discrete(expand = expansion(add = 0.55)) +
  labs(title = "b  Median robustness by disease", x = "Median robustness", y = NULL) +
  theme_nature() +
  theme(legend.position = "none", axis.text.y = element_blank(), axis.ticks.y = element_blank())

# ------------------------------------------------------------
# 9. CONFIGURATION COUNT PANEL
# ------------------------------------------------------------
p_ncfg <- ggplot(n_cfg_df, aes(x = n_configs, y = disease)) +
  geom_col(width = 0.48, fill = "grey35") +
  geom_text(aes(label = n_configs), hjust = -0.18, size = 3.2) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  scale_y_discrete(expand = expansion(add = 0.55)) +
  labs(title = "c  Number of configurations", x = "Configurations", y = NULL) +
  theme_nature() +
  theme(legend.position = "none", axis.text.y = element_blank(), axis.ticks.y = element_blank())

# ------------------------------------------------------------
# 10. COMBINE
# ------------------------------------------------------------
final_plot <- p_main / (p_summary | p_ncfg) +
  plot_layout(heights = c(4.4, 1.35), guides = "collect") &
  theme(legend.position = "top")

print(final_plot)

# ------------------------------------------------------------
# 11. SAVE
# ------------------------------------------------------------
ggsave(
  filename = "Robustness_distributions_across_configurations_high_impact_spaced.pdf",
  plot = final_plot,
  width = 15,
  height = 17,
  units = "in"
)

ggsave(
  filename = "Robustness_distributions_across_configurations_high_impact_spaced.png",
  plot = final_plot,
  width = 15,
  height = 17,
  units = "in",
  dpi = 600,
  bg = "white"
)

ggsave(
  filename = "Robustness_distributions_across_configurations_high_impact_spaced.tiff",
  plot = final_plot,
  width = 15,
  height = 17,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)
