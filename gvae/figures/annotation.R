#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(scales)
  library(glue)
  library(ggrepel)
  library(grid)
  library(cowplot)
})

# ============================================================
# 1. INPUTS
# ============================================================
data_dir <- "~/Documents/Research-Projects/gVAE/docs/"

gene_summary_file <- file.path(data_dir, "combined_figure_gene_heatmap_summary.csv")
path_summary_file <- file.path(data_dir, "combined_figure_pathway_heatmap_summary.csv")

out_prefix <- file.path(
  data_dir,
  "Figure_Disease_Specific_Interpretation_Gene_Pathway_Recovery"
)

if (!file.exists(gene_summary_file)) {
  stop("Missing file: combined_figure_gene_heatmap_summary.csv")
}
if (!file.exists(path_summary_file)) {
  stop("Missing file: combined_figure_pathway_heatmap_summary.csv")
}

# ============================================================
# 2. DISEASE METADATA
# ============================================================
disease_meta <- tribble(
  ~Disease, ~Disease_full, ~Phenotype_type, ~Family, ~Sample_size, ~No_SNPs, ~Proxy_flag, ~Proxy_note,
  "BD",  "Bipolar Disorder",         "Binary",       "Neuropsychiatric",  4806, 392768, FALSE, "",
  "CAD", "Coronary Artery Disease",  "Binary",       "Cardiometabolic",   4864, 392768, FALSE, "",
  "CD",  "Crohn's Disease",          "Binary",       "Immune",            4686, 392768, TRUE,  "UC proxy",
  "T1D", "Type 1 Diabetes",          "Binary",       "Immune",            4901, 391182, FALSE, "",
  "T2D", "Type 2 Diabetes",          "Binary",       "Cardiometabolic",   4862, 392768, FALSE, "",
  "HT",  "Hypertension",             "Binary",       "Cardiometabolic",   4890, 392768, FALSE, "",
  "RA",  "Rheumatoid Arthritis",     "Binary",       "Immune",            4798, 392768, FALSE, "",
  "ALZ", "Alzheimer's Disease",      "Binary",       "Neurodegenerative", 2469, 242983, FALSE, "",
  "ASD", "Autism Spectrum Disorder", "Binary",       "Neurodevelopmental",2279, 639368, FALSE, "",
  "BRC", "Breast Cancer",            "Binary",       "Cancer",           15468, 628989, FALSE, "",
  "COL", "Colon/Colorectal Cancer",  "Binary",       "Cancer",           14688, 623422, FALSE, "",
  "PRC", "Prostate Cancer",          "Binary",       "Cancer",           14657, 623630, FALSE, "",
  "LUN", "Lung Cancer",              "Binary",       "Cancer",           18821, 625120, FALSE, "",
  "BMI", "Body-Mass Index",          "Quantitative", "Quantitative",     10000, 614995, FALSE, "",
  "HDL", "HDL Cholesterol",          "Quantitative", "Quantitative",     10000, 617190, TRUE,  "Metabolic syndrome X proxy",
  "HGT", "Height",                   "Quantitative", "Quantitative",     10000, 614696, TRUE,  "Osteoporosis proxy",
  "LDL", "LDL Cholesterol",          "Quantitative", "Quantitative",     10000, 618972, TRUE,  "Metabolic syndrome X proxy",
  "EOS", "Eosinophil Count",         "Quantitative", "Quantitative",     10000, 614696, TRUE,  "Barrett's esophagus proxy"
)

# ============================================================
# 3. READ SUMMARY FILES
# ============================================================
gene_df <- readr::read_csv(gene_summary_file, show_col_types = FALSE)
path_df <- readr::read_csv(path_summary_file, show_col_types = FALSE)

gene_total_df <- gene_df %>%
  group_by(Disease) %>%
  summarise(
    total_gene_recovery = sum(n_disgenet_genes, na.rm = TRUE),
    mean_gene_signal = mean(heat_value, na.rm = TRUE),
    max_gene_signal = max(heat_value, na.rm = TRUE),
    .groups = "drop"
  )

path_total_df <- path_df %>%
  group_by(Disease) %>%
  summarise(
    total_pathway_recovery = sum(n_configs_hit, na.rm = TRUE),
    mean_path_signal = mean(intensity_capped, na.rm = TRUE),
    max_path_signal = max(intensity_capped, na.rm = TRUE),
    .groups = "drop"
  )

plot_df <- disease_meta %>%
  left_join(gene_total_df, by = "Disease") %>%
  left_join(path_total_df, by = "Disease") %>%
  mutate(
    total_gene_recovery    = replace_na(total_gene_recovery, 0),
    mean_gene_signal       = replace_na(mean_gene_signal, 0),
    max_gene_signal        = replace_na(max_gene_signal, 0),
    total_pathway_recovery = replace_na(total_pathway_recovery, 0),
    mean_path_signal       = replace_na(mean_path_signal, 0),
    max_path_signal        = replace_na(max_path_signal, 0)
  )

# ============================================================
# 4. DERIVED METRICS
# ============================================================
plot_df <- plot_df %>%
  mutate(
    gene_z = as.numeric(scale(total_gene_recovery)),
    path_z = as.numeric(scale(total_pathway_recovery)),
    sample_z = as.numeric(scale(Sample_size)),
    snp_z = as.numeric(scale(No_SNPs)),
    pathway_minus_gene = path_z - gene_z,
    gene_plus_path = gene_z + path_z
  )

plot_df <- plot_df %>%
  mutate(
    Interpretation_class = case_when(
      Disease %in% c("HT","CAD","T2D","RA","BMI","HDL","LDL") ~ "Systemic-shared biology",
      Disease == "ALZ" ~ "Pathway-dominant systems biology",
      Disease == "ASD" ~ "Heterogeneous architecture",
      Disease %in% c("COL","LUN","PRC","BRC") ~ "Tissue-specific / subtype-rich",
      Proxy_flag ~ "Proxy-sensitive interpretation",
      Disease %in% c("T1D","CD") ~ "Immune but context-sensitive",
      TRUE ~ "Mixed architecture"
    ),
    Interpretation_short = case_when(
      Disease == "HT"  ~ "Shared vascular, metabolic, and inflammatory programs.",
      Disease == "CAD" ~ "Shared vascular, metabolic, and inflammatory programs.",
      Disease == "T2D" ~ "Shared metabolic and inflammatory programs.",
      Disease == "RA"  ~ "Systemic immune-inflammatory biology supports direct overlap.",
      Disease == "BMI" ~ "Broad cardiometabolic polygenicity yields distributed recovery.",
      Disease == "HDL" ~ "Stable lipid biology with proxy-sensitive validation.",
      Disease == "LDL" ~ "Stable lipid biology with proxy-sensitive validation.",
      Disease == "HGT" ~ "Broad polygenicity produces stable but diffuse recovery.",
      Disease == "EOS" ~ "Immune / hematologic signal may emerge more at pathway level.",
      Disease == "ALZ" ~ "More coherent at pathway level than at direct gene level.",
      Disease == "ASD" ~ "High etiologic heterogeneity likely dilutes direct overlap.",
      Disease == "BD"  ~ "Neuropsychiatric biology may be more diffuse and mixed.",
      Disease == "COL" ~ "Tissue-specific germline architecture with pathway convergence.",
      Disease == "LUN" ~ "Selective tissue-specific signal with pathway recurrence.",
      Disease == "PRC" ~ "Cancer architecture may be narrower and subtype-dependent.",
      Disease == "BRC" ~ "Cancer architecture may be narrower and subtype-dependent.",
      Disease == "CD"  ~ "Immune biology is present, but validation is proxy-sensitive.",
      Disease == "T1D" ~ "Immune-mediated disease with more context-specific recovery.",
      TRUE ~ "Interpret in the context of architecture and reference alignment."
    )
  )

label_set <- c("HT","CAD","T2D","RA","BMI","HDL","LDL","ALZ","ASD","COL","LUN","CD")
plot_df <- plot_df %>%
  mutate(label_scatter = ifelse(Disease %in% label_set, Disease, NA_character_))

# ============================================================
# 5. PALETTES
# ============================================================
family_palette <- c(
  "Cardiometabolic"    = "#C65D4B",
  "Immune"             = "#8E6CBB",
  "Neurodegenerative"  = "#4B8F8C",
  "Neurodevelopmental" = "#3D5A80",
  "Neuropsychiatric"   = "#6D597A",
  "Cancer"             = "#D17C2F",
  "Quantitative"       = "#4F9D69"
)

interpret_palette <- c(
  "Systemic-shared biology"          = "#C65D4B",
  "Pathway-dominant systems biology" = "#4B8F8C",
  "Heterogeneous architecture"       = "#3D5A80",
  "Tissue-specific / subtype-rich"   = "#D17C2F",
  "Proxy-sensitive interpretation"   = "#B38F2D",
  "Immune but context-sensitive"     = "#8E6CBB",
  "Mixed architecture"               = "#7A7A7A"
)

shape_palette <- c(
  "Binary" = 21,
  "Quantitative" = 24
)

driver_fill_palette <- c(
  "Architecture" = "#F4ECE7",
  "Trait"        = "#E8F2EC",
  "Power"        = "#EAF0F9",
  "Proxy"        = "#F7F2E0"
)

driver_line_palette <- c(
  "Architecture" = "#C7A189",
  "Trait"        = "#7FA68E",
  "Power"        = "#819BC9",
  "Proxy"        = "#B79C3C"
)

panel_d_group_palette <- c(
  "Cardiometabolic" = "#C65D4B",
  "Immune"          = "#8E6CBB",
  "Neuro"           = "#4B7C8F",
  "Cancer"          = "#D17C2F",
  "Quantitative"    = "#4F9D69"
)

# ============================================================
# 6. TYPOGRAPHY + HELPERS
# ============================================================
panel_title_pt      <- 14.8
panel_subtitle_pt   <- 9.8
panel_caption_pt    <- 8.6
panel_card_title_pt <- 8.8
panel_card_body_pt  <- 8.0
panel_header_pt     <- 9.4

wrap_text <- function(x, width = 80) {
  stringr::str_wrap(x, width = width)
}

theme_highimpact <- function() {
  theme_minimal(base_size = 12.5, base_family = "sans") +
    theme(
      panel.grid.major = element_line(colour = "#EBEBEB", linewidth = 0.35),
      panel.grid.minor = element_blank(),
      axis.line = element_line(colour = "black", linewidth = 0.45),
      axis.text = element_text(colour = "black", size = 9.8),
      axis.title = element_text(colour = "black", size = 11.5, face = "bold"),
      plot.title = element_text(size = panel_title_pt, face = "bold", colour = "black", hjust = 0),
      plot.subtitle = element_text(size = panel_subtitle_pt, colour = "#4A4A4A", hjust = 0, lineheight = 1.04),
      plot.caption = element_text(size = panel_caption_pt, colour = "#565656", hjust = 0, lineheight = 1.03),
      legend.title = element_text(size = 10.0, face = "bold"),
      legend.text = element_text(size = 9.1),
      plot.margin = margin(7, 7, 7, 7)
    )
}

make_card_grob <- function(
    title,
    body,
    fill = "#F7F7F7",
    border = "#9A9A9A",
    title_col = "black",
    body_col = "#222222",
    title_wrap = 32,
    body_wrap = 54,
    title_size = panel_card_title_pt,
    body_size = panel_card_body_pt,
    lineheight = 1.02
) {
  title_wrapped <- stringr::str_wrap(title, width = title_wrap)
  body_wrapped  <- stringr::str_wrap(body,  width = body_wrap)
  
  grid::grobTree(
    grid::roundrectGrob(
      x = 0.5, y = 0.5,
      width = 0.992, height = 0.992,
      r = grid::unit(0.035, "snpc"),
      gp = grid::gpar(fill = fill, col = border, lwd = 0.95)
    ),
    grid::roundrectGrob(
      x = 0.5, y = 0.865,
      width = 0.965, height = 0.16,
      r = grid::unit(0.028, "snpc"),
      gp = grid::gpar(fill = scales::alpha(border, 0.08), col = NA)
    ),
    grid::textGrob(
      label = title_wrapped,
      x = grid::unit(0.045, "npc"),
      y = grid::unit(0.90, "npc"),
      just = c("left", "top"),
      gp = grid::gpar(
        col = title_col,
        fontsize = title_size,
        fontface = "bold",
        lineheight = 1.00
      )
    ),
    grid::segmentsGrob(
      x0 = grid::unit(0.04, "npc"),
      x1 = grid::unit(0.96, "npc"),
      y0 = grid::unit(0.71, "npc"),
      y1 = grid::unit(0.71, "npc"),
      gp = grid::gpar(
        col = scales::alpha(border, 0.55),
        lwd = 0.75
      )
    ),
    grid::textGrob(
      label = body_wrapped,
      x = grid::unit(0.045, "npc"),
      y = grid::unit(0.63, "npc"),
      just = c("left", "top"),
      gp = grid::gpar(
        col = body_col,
        fontsize = body_size,
        fontface = "plain",
        lineheight = lineheight
      )
    )
  )
}

make_header_grob <- function(
    label,
    fill = "#4A4A4A",
    text_col = "white",
    border = NULL,
    font_size = panel_header_pt
) {
  if (is.null(border)) border <- fill
  
  grid::grobTree(
    grid::roundrectGrob(
      x = 0.5, y = 0.5,
      width = 0.996, height = 0.985,
      r = grid::unit(0.07, "snpc"),
      gp = grid::gpar(fill = fill, col = border, lwd = 1.0)
    ),
    grid::textGrob(
      label = label,
      x = grid::unit(0.04, "npc"),
      y = grid::unit(0.52, "npc"),
      just = c("left", "center"),
      gp = grid::gpar(
        col = text_col,
        fontsize = font_size,
        fontface = "bold"
      )
    )
  )
}

# ============================================================
# 7. PANEL A — DISEASE RECOVERY MAP
# ============================================================
x_min <- min(plot_df$total_gene_recovery)
x_max <- max(plot_df$total_gene_recovery)
y_min <- min(plot_df$total_pathway_recovery)
y_max <- max(plot_df$total_pathway_recovery)

x_med <- median(plot_df$total_gene_recovery, na.rm = TRUE)
y_med <- median(plot_df$total_pathway_recovery, na.rm = TRUE)

set.seed(123)

p_scatter <- ggplot(
  plot_df,
  aes(
    x = total_gene_recovery,
    y = total_pathway_recovery,
    fill = Family,
    shape = Phenotype_type,
    size = Sample_size
  )
) +
  geom_point(
    colour = "black",
    stroke = 0.45,
    alpha = 0.94
  ) +
  ggrepel::geom_text_repel(
    data = dplyr::filter(plot_df, !is.na(label_scatter)),
    aes(label = label_scatter),
    size = 3.8,
    fontface = "bold",
    colour = "#1F1F1F",
    box.padding = 0.45,
    point.padding = 0.30,
    segment.color = "#6E6E6E",
    segment.size = 0.35,
    min.segment.length = 0,
    max.overlaps = Inf,
    seed = 123,
    show.legend = FALSE
  ) +
  geom_vline(
    xintercept = x_med,
    linetype = "dashed",
    colour = "#A7A7A7",
    linewidth = 0.45
  ) +
  geom_hline(
    yintercept = y_med,
    linetype = "dashed",
    colour = "#A7A7A7",
    linewidth = 0.45
  ) +
  annotate(
    "text",
    x = x_min + 0.06 * (x_max - x_min),
    y = y_max - 0.06 * (y_max - y_min),
    label = "Pathway-dominant\nsystems-level recovery",
    hjust = 0,
    vjust = 1,
    size = 3.45,
    colour = "#5A5A5A",
    fontface = "bold"
  ) +
  annotate(
    "text",
    x = x_max - 0.04 * (x_max - x_min),
    y = y_max - 0.06 * (y_max - y_min),
    label = "Strong gene + pathway\nrecovery",
    hjust = 1,
    vjust = 1,
    size = 3.45,
    colour = "#5A5A5A",
    fontface = "bold"
  ) +
  annotate(
    "text",
    x = x_max - 0.04 * (x_max - x_min),
    y = y_min + 0.08 * (y_max - y_min),
    label = "Gene-richer,\npathway-moderate",
    hjust = 1,
    vjust = 0,
    size = 3.45,
    colour = "#5A5A5A",
    fontface = "bold"
  ) +
  annotate(
    "text",
    x = x_min + 0.06 * (x_max - x_min),
    y = y_min + 0.08 * (y_max - y_min),
    label = "Narrower / heterogeneous /\nproxy-sensitive",
    hjust = 0,
    vjust = 0,
    size = 3.45,
    colour = "#5A5A5A",
    fontface = "bold"
  ) +
  scale_fill_manual(values = family_palette) +
  scale_shape_manual(values = shape_palette) +
  scale_size_continuous(
    range = c(4.5, 11.5),
    labels = label_comma()
  ) +
  guides(
    fill = guide_legend(order = 1, override.aes = list(size = 5, shape = 21)),
    shape = guide_legend(order = 2),
    size = guide_legend(order = 3)
  ) +
  labs(
    title = "a  Disease-specific recovery map",
    subtitle = wrap_text(
      "Each point represents one disease or trait. Position reflects joint recovery at the direct gene level and the pathway level; point size indicates cohort size.",
      88
    ),
    x = "Total disease-relevant genes captured",
    y = "Total disease-relevant pathways captured",
    fill = "Phenotype family",
    shape = "Phenotype type",
    size = "Sample size",
    caption = wrap_text(
      "Dashed lines indicate the median recovery across diseases. Diseases in the upper-right combine stronger direct gene recovery with stronger pathway recurrence, whereas diseases shifted upward relative to their gene recovery are relatively more pathway-dominant.",
      115
    )
  ) +
  coord_cartesian(clip = "off") +
  theme_highimpact() +
  theme(
    legend.position = "right"
  )

# ============================================================
# 8. PANEL B — DISEASE RECOVERY PROFILES
# ============================================================
disease_rank_df <- plot_df %>%
  mutate(
    total_pathway_recovery_scaled = scales::rescale(
      total_pathway_recovery,
      to = range(total_gene_recovery, na.rm = TRUE)
    )
  ) %>%
  arrange(desc(total_pathway_recovery), desc(total_gene_recovery)) %>%
  mutate(
    Disease = factor(Disease, levels = rev(Disease))
  )

p_rank <- ggplot(disease_rank_df, aes(y = Disease)) +
  geom_segment(
    aes(
      x = total_pathway_recovery_scaled,
      xend = total_gene_recovery,
      yend = Disease
    ),
    colour = "#D1D1D1",
    linewidth = 1.0
  ) +
  geom_point(
    aes(x = total_gene_recovery, fill = Interpretation_class),
    shape = 21,
    colour = "black",
    stroke = 0.35,
    size = 3.8,
    show.legend = FALSE
  ) +
  geom_point(
    aes(x = total_pathway_recovery_scaled, colour = Interpretation_class),
    size = 3.1,
    show.legend = FALSE
  ) +
  scale_fill_manual(values = interpret_palette, guide = "none") +
  scale_colour_manual(values = interpret_palette, guide = "none") +
  scale_x_continuous(
    expand = expansion(mult = c(0.02, 0.08)),
    labels = label_comma()
  ) +
  labs(
    title = "b  Disease-specific recovery profiles",
    subtitle = wrap_text(
      "For each disease, the filled marker shows total gene-level recovery and the smaller colored marker shows pathway recovery rescaled to the same axis for comparison.",
      88
    ),
    x = "Relative recovery scale",
    y = NULL
  ) +
  theme_highimpact() +
  theme(
    axis.text.y = element_text(face = "bold", size = 9.6),
    axis.ticks.y = element_blank(),
    legend.position = "none"
  )

# ============================================================
# 9. PANEL C — COMPACT 2x2 DRIVER CARDS + EMBEDDED FIGURE CAPTION
# ============================================================
driver_cards <- tribble(
  ~Driver, ~Body, ~Group,
  "Disease architecture",
  "Shared systemic phenotypes often show stronger direct gene recovery because vascular, metabolic, immune, and inflammatory programs are reused across related diseases.",
  "Architecture",
  
  "Trait type",
  "Quantitative traits often yield smoother distributed signal, whereas heterogeneous binary phenotypes can dilute direct overlap while preserving broader pathway structure.",
  "Trait",
  
  "Sample size / power",
  "Smaller cohorts and reduced SNP coverage can weaken direct recovery, especially for biologically diffuse diseases whose effects are spread across many loci and pathways.",
  "Power",
  
  "Ontology / proxy alignment",
  "Weaker overlap can partly reflect imperfect validation-reference alignment, particularly when proxy disease terms are used in DisGeNET-based evaluation.",
  "Proxy"
)

panel_c_caption <- wrap_text(
  paste0(
    "Biological observations: HT, CAD, T2D, RA, BMI, HDL, and LDL are consistent with shared systemic ",
    "cardiometabolic-inflammatory biology; ALZ appears more coherent at the pathway level than in direct ",
    "disease-gene overlap; ASD likely reflects etiologic heterogeneity and smaller-sample diffusion of signal; ",
    "COL and LUN illustrate tissue-specific phenotypes that may preserve pathway convergence despite narrower ",
    "direct overlap; proxy-mapped diseases should be interpreted cautiously because validation-reference mismatch ",
    "can reduce apparent overlap."
  ),
  118
)

p_strip <- cowplot::ggdraw() +
  cowplot::draw_label(
    "c  Biological drivers of disease-specific behavior",
    x = 0.00, y = 0.995,
    hjust = 0, vjust = 1,
    fontface = "bold",
    size = panel_title_pt
  ) +
  cowplot::draw_label(
    wrap_text(
      "These four axes provide the main biological rationale for why diseases occupy different regions of the recovery map.",
      108
    ),
    x = 0.00, y = 0.965,
    hjust = 0, vjust = 1,
    size = panel_subtitle_pt,
    colour = "#4A4A4A",
    lineheight = 1.02
  )

# tighter card layout with reserved footer area for figure caption
driver_positions <- tribble(
  ~Driver, ~x,    ~y,   ~w,    ~h,
  "Disease architecture",        0.00, 0.57, 0.49, 0.20,
  "Trait type",                  0.51, 0.57, 0.49, 0.20,
  "Sample size / power",         0.00, 0.32, 0.49, 0.20,
  "Ontology / proxy alignment",  0.51, 0.32, 0.49, 0.20
)

driver_plot_df <- driver_cards %>%
  left_join(driver_positions, by = "Driver")

for (i in seq_len(nrow(driver_plot_df))) {
  row_i <- driver_plot_df[i, ]
  
  p_strip <- p_strip +
    cowplot::draw_grob(
      make_card_grob(
        title = row_i$Driver,
        body  = row_i$Body,
        fill  = driver_fill_palette[[row_i$Group]],
        border = driver_line_palette[[row_i$Group]],
        title_wrap = 28,
        body_wrap = 42,
        title_size = 9.2,
        body_size = 8.3,
        lineheight = 1.03
      ),
      x = row_i$x, y = row_i$y, width = row_i$w, height = row_i$h
    )
}

# elegant divider above the embedded figure caption
p_strip <- p_strip +
  cowplot::draw_grob(
    grid::segmentsGrob(
      x0 = grid::unit(0.00, "npc"),
      x1 = grid::unit(1.00, "npc"),
      y0 = grid::unit(0.21, "npc"),
      y1 = grid::unit(0.21, "npc"),
      gp = grid::gpar(col = "#CFCFCF", lwd = 0.9)
    ),
    x = 0, y = 0, width = 1, height = 1
  ) +
  cowplot::draw_label(
    panel_c_caption,
    x = 0.00, y = 0.18,
    hjust = 0, vjust = 1,
    size = panel_caption_pt,
    colour = "#565656",
    lineheight = 1.04
  )

# ============================================================
# 10. PANEL D — WIDER PANEL + TALLER NON-OVERLAPPING CARDS
# ============================================================
panel_d_df <- plot_df %>%
  mutate(
    PanelD_group = case_when(
      Disease %in% c("HT","CAD","T2D")               ~ "Cardiometabolic",
      Disease %in% c("RA","CD","T1D")                ~ "Immune",
      Disease %in% c("ALZ","ASD","BD")               ~ "Neuro",
      Disease %in% c("COL","LUN","PRC","BRC")        ~ "Cancer",
      Disease %in% c("BMI","HDL","LDL","HGT","EOS")  ~ "Quantitative",
      TRUE ~ "Other"
    ),
    Disease_short = case_when(
      Disease == "HT"  ~ "Hypertension",
      Disease == "CAD" ~ "Coronary artery disease",
      Disease == "T2D" ~ "Type 2 diabetes",
      Disease == "RA"  ~ "Rheumatoid arthritis",
      Disease == "CD"  ~ "Crohn's disease",
      Disease == "T1D" ~ "Type 1 diabetes",
      Disease == "ALZ" ~ "Alzheimer's disease",
      Disease == "ASD" ~ "Autism spectrum disorder",
      Disease == "BD"  ~ "Bipolar disorder",
      Disease == "COL" ~ "Colorectal cancer",
      Disease == "LUN" ~ "Lung cancer",
      Disease == "PRC" ~ "Prostate cancer",
      Disease == "BRC" ~ "Breast cancer",
      Disease == "BMI" ~ "Body-mass index",
      Disease == "HDL" ~ "HDL cholesterol",
      Disease == "LDL" ~ "LDL cholesterol",
      Disease == "HGT" ~ "Height",
      Disease == "EOS" ~ "Eosinophil count",
      TRUE ~ Disease_full
    ),
    Card_title = glue("{Disease} | {Disease_short}"),
    Card_body = case_when(
      Disease == "HT"  ~ "Shared vascular, metabolic, and inflammatory programs.",
      Disease == "CAD" ~ "Shared vascular, metabolic, and inflammatory programs.",
      Disease == "T2D" ~ "Shared metabolic-inflammatory programs support direct overlap.",
      Disease == "RA"  ~ "Systemic immune-inflammatory biology supports direct overlap.",
      Disease == "CD"  ~ "Immune signal is present; validation is proxy-sensitive (UC proxy).",
      Disease == "T1D" ~ "Immune-mediated disease with more context-specific recovery.",
      Disease == "ALZ" ~ "Pathway-level coherence exceeds direct gene overlap.",
      Disease == "ASD" ~ "Etiologic heterogeneity likely dilutes direct overlap.",
      Disease == "BD"  ~ "Neuropsychiatric biology appears more diffuse and mixed.",
      Disease == "COL" ~ "Tissue-specific germline architecture with pathway convergence.",
      Disease == "LUN" ~ "Selective tissue-specific signal with pathway recurrence.",
      Disease == "PRC" ~ "Cancer signal may be narrower and subtype-dependent.",
      Disease == "BRC" ~ "Cancer signal may be narrower and subtype-dependent.",
      Disease == "BMI" ~ "Broad cardiometabolic polygenicity yields distributed recovery.",
      Disease == "HDL" ~ "Stable lipid biology; evaluation is proxy-sensitive.",
      Disease == "LDL" ~ "Stable lipid biology; evaluation is proxy-sensitive.",
      Disease == "HGT" ~ "Diffuse polygenic recovery; validation uses an osteoporosis proxy.",
      Disease == "EOS" ~ "Pathway-level immune/hematologic signal; Barrett's proxy in validation.",
      TRUE ~ Interpretation_short
    )
  ) %>%
  select(Disease, Disease_full, PanelD_group, Card_title, Card_body)

panel_d_order <- c(
  "HT","CAD","T2D",
  "RA","CD","T1D",
  "ALZ","ASD","BD",
  "COL","LUN","PRC","BRC",
  "BMI","HDL","LDL","HGT","EOS"
)

panel_d_df <- panel_d_df %>%
  mutate(Disease = factor(Disease, levels = panel_d_order)) %>%
  arrange(Disease)

left_group_order  <- c("Cardiometabolic", "Immune", "Neuro")
right_group_order <- c("Cancer", "Quantitative")

group_diseases <- list(
  "Cardiometabolic" = c("HT","CAD","T2D"),
  "Immune"          = c("RA","CD","T1D"),
  "Neuro"           = c("ALZ","ASD","BD"),
  "Cancer"          = c("COL","LUN","PRC","BRC"),
  "Quantitative"    = c("BMI","HDL","LDL","HGT","EOS")
)

panel_d_lookup <- panel_d_df %>%
  mutate(Disease = as.character(Disease))

add_family_section <- function(
    p,
    family_name,
    disease_ids,
    x_left,
    y_top,
    section_width,
    header_h = 0.040,
    card_h = 0.082,
    gap = 0.008,
    section_gap = 0.012
) {
  family_col <- panel_d_group_palette[[family_name]]
  
  p <- p + cowplot::draw_grob(
    make_header_grob(
      label = family_name,
      fill = family_col,
      text_col = "white",
      border = family_col,
      font_size = panel_header_pt
    ),
    x = x_left,
    y = y_top - header_h,
    width = section_width,
    height = header_h
  )
  
  y_cursor <- y_top - header_h - gap
  
  for (d in disease_ids) {
    row_i <- panel_d_lookup %>% filter(Disease == d)
    
    p <- p + cowplot::draw_grob(
      make_card_grob(
        title = row_i$Card_title,
        body  = row_i$Card_body,
        fill  = scales::alpha(family_col, 0.13),
        border = family_col,
        title_wrap = 30,
        body_wrap = 48,
        title_size = 8.9,
        body_size = 8.1,
        lineheight = 1.03
      ),
      x = x_left,
      y = y_cursor - card_h,
      width = section_width,
      height = card_h
    )
    
    y_cursor <- y_cursor - card_h - gap
  }
  
  y_cursor <- y_cursor - section_gap
  list(plot = p, y_cursor = y_cursor)
}

p_callout <- cowplot::ggdraw() +
  cowplot::draw_label(
    "d  Disease-specific biological interpretations",
    x = 0.00, y = 0.995,
    hjust = 0, vjust = 1,
    fontface = "bold",
    size = panel_title_pt
  ) +
  cowplot::draw_label(
    wrap_text(
      "Rounded interpretation cards grouped by disease family. These annotations provide a concise biological reading of the observed gene- and pathway-level recovery.",
      122
    ),
    x = 0.00, y = 0.965,
    hjust = 0, vjust = 1,
    size = panel_subtitle_pt,
    colour = "#4A4A4A",
    lineheight = 1.02
  )

left_x  <- 0.000
right_x <- 0.507
col_w   <- 0.488
start_y <- 0.89

y_left <- start_y
for (grp in left_group_order) {
  res <- add_family_section(
    p = p_callout,
    family_name = grp,
    disease_ids = group_diseases[[grp]],
    x_left = left_x,
    y_top = y_left,
    section_width = col_w
  )
  p_callout <- res$plot
  y_left <- res$y_cursor
}

y_right <- start_y
for (grp in right_group_order) {
  res <- add_family_section(
    p = p_callout,
    family_name = grp,
    disease_ids = group_diseases[[grp]],
    x_left = right_x,
    y_top = y_right,
    section_width = col_w
  )
  p_callout <- res$plot
  y_right <- res$y_cursor
}

# ============================================================
# 11. GLOBAL CAPTION AS A FULL-WIDTH PANEL
# ============================================================
main_title <- "Disease-specific interpretation of gene- and pathway-level recovery"

main_subtitle <- wrap_text(
  paste0(
    "Disease-specific differences in gVAE biological recovery likely reflect a combination of phenotype architecture, ",
    "trait type, sample size / SNP coverage, and validation-reference alignment. ",
    "Systemic cardiometabolic and inflammatory phenotypes tend to show stronger direct gene-level recovery, ",
    "whereas heterogeneous or diffuse diseases more often emerge through pathway-level convergence."
  ),
  175
)

# ============================================================
# 12. COMBINE PANELS
# ============================================================
top_row <- patchwork::wrap_plots(
  p_scatter, p_rank,
  ncol = 2,
  widths = c(0.86, 1.14)
)

bottom_row <- patchwork::wrap_plots(
  p_strip, p_callout,
  ncol = 2,
  widths = c(0.76, 1.24)
)

main_body <- patchwork::wrap_plots(
  top_row, bottom_row,
  ncol = 1,
  heights = c(0.88, 1.12)
)

final_plot <- patchwork::wrap_plots(
  main_body, 
  ncol = 1,
  heights = c(0.965, 0.035)
) +
  plot_annotation(
    title = main_title,
    subtitle = main_subtitle,
    theme = theme(
      plot.title = element_text(size = 18.2, face = "bold", hjust = 0, margin = margin(b = 4)),
      plot.subtitle = element_text(size = 10.4, colour = "#444444", hjust = 0, lineheight = 1.08, margin = margin(b = 8))
    )
  ) &
  theme(
    plot.background = element_rect(fill = "white", colour = NA)
  )

# ============================================================
# 13. DISPLAY
# ============================================================
print(final_plot)

# ============================================================
# 14. SAVE
# ============================================================
ggsave(
  filename = paste0(out_prefix, ".pdf"),
  plot = final_plot,
  width = 19.6,
  height = 15.4,
  units = "in",
  dpi = 600,
  device = "pdf",
  bg = "white"
)

ggsave(
  filename = paste0(out_prefix, ".png"),
  plot = final_plot,
  width = 19.6,
  height = 15.4,
  units = "in",
  dpi = 600,
  bg = "white"
)

ggsave(
  filename = paste0(out_prefix, ".tiff"),
  plot = final_plot,
  width = 19.6,
  height = 15.4,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)

# ============================================================
# 15. EXPORT TABLE
# ============================================================
readr::write_csv(
  plot_df %>%
    select(
      Disease, Disease_full, Family, Phenotype_type,
      Sample_size, No_SNPs, Proxy_flag, Proxy_note,
      total_gene_recovery, total_pathway_recovery,
      mean_gene_signal, mean_path_signal,
      gene_z, path_z, pathway_minus_gene, gene_plus_path,
      Interpretation_class, Interpretation_short
    ),
  file.path(data_dir, "disease_specific_interpretation_figure_table_v7.csv")
)
