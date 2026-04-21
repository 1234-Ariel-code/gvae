library(tidyverse)
library(patchwork)
library(viridis)
library(scales)
library(glue)
library(cowplot)
library(grid)

# ------------------------------------------------------------
# 1. GLOBAL INPUTS
# ------------------------------------------------------------
data_dir <- "~/Documents/Research-Projects/gVAE/docs/"

gene_wide_pattern <- "^latent_gene_table\\.ALL_SAMPLES\\.wide_gene_x_sampleLV_.*\\.csv$"
disgenet_file <- file.path(data_dir, "consolidated.tsv")
pathway_file_pattern <- "^overlap_terms_by_library_topN_[A-Za-z0-9]+_LD[0-9]+_NS[0-9]+_L[0-9]+\\.csv$"

out_prefix_clean <- file.path(data_dir, "Figure_Gene_and_Pathway_Relevance")
out_prefix_annotated <- file.path(data_dir, "Figure_Gene_and_Pathway_Relevance")

disease_codes <- c(
  "ALZ","ASD","BD","BMI","BRC","CAD","CD","COL","EOS",
  "HDL","HGT","HT","LDL","LUN","PRC","RA","T1D","T2D"
)

# ---- Gene controls
USE_SQRT_TRANSFORM_GENE <- TRUE

# ---- Pathway controls
FDR_THRESHOLD <- 0.05
USE_NEGLOG10_FDR <- TRUE
CAP_INTENSITY_AT_QUANTILE <- 0.99
MAX_PATHWAYS_TO_SHOW <- 18
MIN_DISEASE_SUPPORT_FOR_PATHWAY <- 1
TRUNCATE_LABEL_TO <- 24

# ---- Distinct palettes
GENE_VIRIDIS_OPTION <- "magma"
PATHWAY_VIRIDIS_OPTION <- "cividis"

# ------------------------------------------------------------
# 2. THEME
# ------------------------------------------------------------
theme_journal <- function() {
  theme_minimal(base_size = 12.5, base_family = "sans") +
    theme(
      panel.grid = element_blank(),
      axis.line = element_line(colour = "black", linewidth = 0.45),
      axis.text = element_text(colour = "black", size = 9.8),
      axis.title = element_text(colour = "black", size = 11.5, face = "bold"),
      plot.title = element_text(size = 14.2, face = "bold", hjust = 0, margin = margin(b = 2)),
      plot.subtitle = element_text(size = 9.8, colour = "grey30", hjust = 0, lineheight = 1.02, margin = margin(b = 4)),
      plot.caption = element_text(size = 8.6, colour = "grey30", hjust = 0, lineheight = 1.0, margin = margin(t = 4)),
      legend.title = element_text(size = 10.2, face = "bold"),
      legend.text = element_text(size = 9.5),
      plot.margin = margin(3, 3, 3, 3)
    )
}

# ============================================================
# 3. GENE ANALYSIS
# ============================================================

disease_map <- tribble(
  ~Disease, ~DisGeNET_term,                ~Mapping_note,
  "ALZ",    "Alzheimer's disease",         "exact",
  "ASD",    "autistic disorder",           "exact",
  "BD",     "bipolar disorder",            "exact",
  "BMI",    "obesity",                     "proxy for BMI trait",
  "BRC",    "breast cancer",               "exact",
  "CAD",    "coronary artery disease",     "exact",
  "CD",     "ulcerative colitis",          "proxy",
  "COL",    "colon cancer",                "proxy",
  "EOS",    "Barrett's esophagus",         "proxy",
  "HDL",    "metabolic syndrome X",        "proxy",
  "HGT",    "osteoporosis",                "proxy",
  "HT",     "hypertension",                "exact",
  "LDL",    "metabolic syndrome X",        "proxy",
  "LUN",    "lung cancer",                 "exact",
  "PRC",    "prostate cancer",             "exact",
  "RA",     "rheumatoid arthritis",        "exact",
  "T1D",    "type 1 diabetes mellitus",    "exact",
  "T2D",    "type 2 diabetes mellitus",    "exact"
)

# ------------------------------------------------------------
# Robust aggregation controls
# ------------------------------------------------------------
MIN_SUPPORT_DRAWS <- 1
MIN_SUPPORT_FRACTION <- 0.10
USE_COUNT_WEIGHTING <- TRUE
USE_SUPPORT_WEIGHTING <- TRUE

gene_wide_files <- list.files(
  path = data_dir,
  pattern = gene_wide_pattern,
  full.names = TRUE
)

if (length(gene_wide_files) == 0) {
  stop("No latent_gene_table.ALL_SAMPLES.wide_gene_x_sampleLV_*.csv files found in data_dir")
}
if (!file.exists(disgenet_file)) {
  stop("consolidated.tsv not found in data_dir")
}

# ------------------------------------------------------------
# DisGeNET scores
# ------------------------------------------------------------
disgenet_raw <- readr::read_tsv(disgenet_file, show_col_types = FALSE) %>%
  dplyr::transmute(
    doid_name = trimws(as.character(doid_name)),
    geneSymbol = toupper(trimws(as.character(geneSymbol))),
    score_max = suppressWarnings(as.numeric(score_max)),
    score_mean = suppressWarnings(as.numeric(score_mean))
  ) %>%
  dplyr::filter(!is.na(doid_name), !is.na(geneSymbol), geneSymbol != "")

disgenet_gene_scores <- disgenet_raw %>%
  dplyr::mutate(best_score = pmax(score_max, score_mean, na.rm = TRUE)) %>%
  dplyr::group_by(doid_name, geneSymbol) %>%
  dplyr::summarise(best_score = max(best_score, na.rm = TRUE), .groups = "drop")

disease_gene_scores <- disease_map %>%
  dplyr::inner_join(
    disgenet_gene_scores,
    by = c("DisGeNET_term" = "doid_name"),
    relationship = "many-to-many"
  )

disease_ref_totals <- disease_gene_scores %>%
  dplyr::group_by(Disease) %>%
  dplyr::summarise(
    total_disgenet_genes_for_disease = dplyr::n_distinct(geneSymbol),
    .groups = "drop"
  )

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
extract_gene_disease_from_filename <- function(path) {
  nm <- basename(path)
  sub("^latent_gene_table\\.ALL_SAMPLES\\.wide_gene_x_sampleLV_([A-Za-z0-9]+)_LD.*$", "\\1", nm)
}

extract_gene_config_from_filename <- function(path) {
  nm <- basename(path)
  cfg <- stringr::str_extract(nm, "LD[0-9]+_NS[0-9]+_L[0-9]+")
  ifelse(is.na(cfg), "UNKNOWN_CFG", cfg)
}

extract_lv_from_samplelv_col <- function(x) {
  lv_num <- stringr::str_extract(x, "(?<=__LD_)\\d+")
  lv_num <- suppressWarnings(as.integer(lv_num))
  lv_label <- ifelse(is.na(lv_num), NA_character_, paste0("LV", lv_num + 1))
  
  sample_id <- stringr::str_extract(x, "^S\\d+")
  tibble(
    sampleLV_col = x,
    Sample = sample_id,
    LV_num = lv_num,
    LV_label = lv_label
  )
}

# ------------------------------------------------------------
# Read wide gene × sampleLV files
# ------------------------------------------------------------
gene_wide_all <- purrr::map_dfr(gene_wide_files, function(f) {
  disease <- extract_gene_disease_from_filename(f)
  config <- extract_gene_config_from_filename(f)
  
  df <- readr::read_csv(f, show_col_types = FALSE)
  
  if (!("GENE" %in% names(df))) {
    stop(glue("GENE column not found in file: {basename(f)}"))
  }
  
  value_cols <- setdiff(names(df), "GENE")
  col_map <- extract_lv_from_samplelv_col(value_cols)
  
  df %>%
    dplyr::mutate(
      Disease = disease,
      Config = config,
      GENE = toupper(trimws(as.character(GENE)))
    ) %>%
    tidyr::pivot_longer(
      cols = all_of(value_cols),
      names_to = "sampleLV_col",
      values_to = "gene_count"
    ) %>%
    dplyr::left_join(col_map, by = "sampleLV_col") %>%
    dplyr::mutate(
      gene_count = suppressWarnings(as.numeric(gene_count))
    ) %>%
    dplyr::filter(
      !is.na(GENE), GENE != "",
      !is.na(LV_num), !is.na(LV_label),
      !is.na(Sample)
    )
})

# ------------------------------------------------------------
# Join disease mapping + DisGeNET
# ------------------------------------------------------------
latent_scored <- gene_wide_all %>%
  dplyr::inner_join(
    disease_map %>% dplyr::select(Disease, DisGeNET_term),
    by = "Disease"
  ) %>%
  dplyr::left_join(
    disease_gene_scores %>% dplyr::select(Disease, geneSymbol, best_score),
    by = c("Disease", "GENE" = "geneSymbol")
  )

# ------------------------------------------------------------
# Support aggregation
# ------------------------------------------------------------
lv_gene_support <- latent_scored %>%
  dplyr::group_by(Disease, Config, LV_num, LV_label, GENE, best_score) %>%
  dplyr::summarise(
    n_draws_total = dplyr::n_distinct(Sample),
    n_support_draws = sum(gene_count > 0, na.rm = TRUE),
    support_fraction = ifelse(n_draws_total > 0, n_support_draws / n_draws_total, 0),
    sum_gene_count = sum(gene_count, na.rm = TRUE),
    mean_gene_count = mean(gene_count, na.rm = TRUE),
    max_gene_count = max(gene_count, na.rm = TRUE),
    .groups = "drop"
  )

lv_gene_supported <- lv_gene_support %>%
  dplyr::mutate(
    pass_support = (n_support_draws >= MIN_SUPPORT_DRAWS) &
      (support_fraction >= MIN_SUPPORT_FRACTION)
  ) %>%
  dplyr::filter(pass_support)

lv_gene_supported <- lv_gene_supported %>%
  dplyr::mutate(
    support_weight = if (USE_SUPPORT_WEIGHTING) support_fraction else 1,
    count_weight   = if (USE_COUNT_WEIGHTING) log1p(sum_gene_count) else 1,
    weighted_gene_score = best_score * support_weight * count_weight
  )

# ------------------------------------------------------------
# Gene diagnostics
# ------------------------------------------------------------
gene_membership_df <- lv_gene_supported %>%
  dplyr::filter(!is.na(best_score)) %>%
  dplyr::group_by(Disease, Config, LV_num, LV_label) %>%
  dplyr::summarise(
    genes = paste(sort(unique(GENE)), collapse = "; "),
    n_genes = dplyr::n_distinct(GENE),
    mean_best_score = mean(best_score, na.rm = TRUE),
    total_best_score = sum(best_score, na.rm = TRUE),
    mean_support_fraction = mean(support_fraction, na.rm = TRUE),
    total_gene_count = sum(sum_gene_count, na.rm = TRUE),
    .groups = "drop"
  )

readr::write_csv(
  gene_membership_df,
  file.path(data_dir, "combined_figure_gene_membership_by_disease_lv.csv")
)

compute_jaccard <- function(a, b) {
  a <- unique(a)
  b <- unique(b)
  inter <- length(intersect(a, b))
  union <- length(union(a, b))
  if (union == 0) return(NA_real_)
  inter / union
}

gene_overlap_df <- lv_gene_supported %>%
  dplyr::filter(!is.na(best_score)) %>%
  dplyr::group_by(Disease, Config, LV_label) %>%
  dplyr::summarise(genes = list(sort(unique(GENE))), .groups = "drop") %>%
  dplyr::group_by(Disease, Config) %>%
  dplyr::group_modify(~{
    df <- .
    if (nrow(df) < 2) return(tibble())
    pairs <- t(combn(seq_len(nrow(df)), 2))
    tibble(
      LV1 = df$LV_label[pairs[,1]],
      LV2 = df$LV_label[pairs[,2]],
      n_genes_lv1 = lengths(df$genes[pairs[,1]]),
      n_genes_lv2 = lengths(df$genes[pairs[,2]]),
      n_overlap = purrr::map2_int(df$genes[pairs[,1]], df$genes[pairs[,2]], ~length(intersect(.x, .y))),
      jaccard = purrr::map2_dbl(df$genes[pairs[,1]], df$genes[pairs[,2]], compute_jaccard)
    )
  }) %>%
  dplyr::ungroup()

readr::write_csv(
  gene_overlap_df,
  file.path(data_dir, "combined_figure_gene_overlap_between_lvs_within_disease.csv")
)

gene_overlap_summary <- gene_overlap_df %>%
  dplyr::group_by(Disease, Config) %>%
  dplyr::summarise(
    mean_jaccard = mean(jaccard, na.rm = TRUE),
    max_jaccard = max(jaccard, na.rm = TRUE),
    mean_overlap = mean(n_overlap, na.rm = TRUE),
    .groups = "drop"
  )

readr::write_csv(
  gene_overlap_summary,
  file.path(data_dir, "combined_figure_gene_overlap_summary_by_disease.csv")
)

# ------------------------------------------------------------
# Gene summarization
# ------------------------------------------------------------
lv_summary_raw <- lv_gene_supported %>%
  dplyr::group_by(Disease, Config, LV_num, LV_label) %>%
  dplyr::summarise(
    n_disgenet_genes = dplyr::n_distinct(GENE),
    weighted_signal = sum(weighted_gene_score, na.rm = TRUE),
    mean_signal = mean(weighted_gene_score, na.rm = TRUE),
    mean_support_fraction = mean(support_fraction, na.rm = TRUE),
    total_gene_count = sum(sum_gene_count, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::mutate(
    n_disgenet_genes = tidyr::replace_na(n_disgenet_genes, 0),
    weighted_signal = tidyr::replace_na(weighted_signal, 0),
    mean_signal = tidyr::replace_na(mean_signal, 0),
    mean_support_fraction = tidyr::replace_na(mean_support_fraction, 0),
    total_gene_count = tidyr::replace_na(total_gene_count, 0)
  )

all_configs_by_disease <- gene_wide_all %>%
  dplyr::distinct(Disease, Config)

all_lv_df <- gene_wide_all %>%
  dplyr::distinct(LV_num, LV_label)

full_lv_grid <- all_configs_by_disease %>%
  tidyr::crossing(all_lv_df)

lv_summary <- full_lv_grid %>%
  dplyr::left_join(
    lv_summary_raw,
    by = c("Disease", "Config", "LV_num", "LV_label")
  ) %>%
  dplyr::mutate(
    n_disgenet_genes = tidyr::replace_na(n_disgenet_genes, 0),
    weighted_signal = tidyr::replace_na(weighted_signal, 0),
    mean_signal = tidyr::replace_na(mean_signal, 0),
    mean_support_fraction = tidyr::replace_na(mean_support_fraction, 0),
    total_gene_count = tidyr::replace_na(total_gene_count, 0)
  ) %>%
  dplyr::left_join(disease_ref_totals, by = "Disease")

gene_heat_df <- lv_summary %>%
  dplyr::mutate(
    capture_fraction = ifelse(
      total_disgenet_genes_for_disease > 0,
      n_disgenet_genes / total_disgenet_genes_for_disease,
      0
    ),
    weighted_fraction = ifelse(
      total_disgenet_genes_for_disease > 0,
      weighted_signal / total_disgenet_genes_for_disease,
      0
    ),
    heat_value = if (USE_SQRT_TRANSFORM_GENE) sqrt(weighted_fraction) else weighted_fraction
  )

# ------------------------------------------------------------
# Gene ordering
# ------------------------------------------------------------
gene_heat_mat <- gene_heat_df %>%
  dplyr::select(Disease, LV_label, heat_value) %>%
  tidyr::pivot_wider(names_from = LV_label, values_from = heat_value) %>%
  dplyr::arrange(Disease)

gene_row_names <- gene_heat_mat$Disease
gene_heat_mat_num <- gene_heat_mat %>%
  dplyr::select(-Disease) %>%
  as.matrix()

rownames(gene_heat_mat_num) <- gene_row_names

gene_col_hc <- stats::hclust(stats::dist(t(gene_heat_mat_num)), method = "ward.D2")
gene_col_order <- colnames(gene_heat_mat_num)[gene_col_hc$order]

gene_col_avg <- colMeans(gene_heat_mat_num[, gene_col_order, drop = FALSE], na.rm = TRUE)
if (mean(gene_col_avg[1:min(3, length(gene_col_avg))], na.rm = TRUE) >
    mean(tail(gene_col_avg, min(3, length(gene_col_avg))), na.rm = TRUE)) {
  gene_col_order <- gene_col_order
}

gene_row_score_df <- gene_heat_df %>%
  dplyr::group_by(Disease) %>%
  dplyr::summarise(
    avg_heat = mean(heat_value, na.rm = TRUE),
    max_heat = max(heat_value, na.rm = TRUE),
    total_heat = sum(heat_value, na.rm = TRUE),
    total_genes = sum(n_disgenet_genes, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(avg_heat, max_heat, total_heat, total_genes)

gene_row_order <- gene_row_score_df$Disease
gene_row_order <- c("ASD", setdiff(gene_row_order, "ASD"))
gene_row_display <- gene_row_order

gene_heat_df <- gene_heat_df %>%
  dplyr::mutate(
    Disease = factor(Disease, levels = gene_row_order),
    LV_label = factor(LV_label, levels = gene_col_order)
  )

gene_peak_df <- gene_heat_df %>%
  dplyr::group_by(Disease) %>%
  dplyr::slice_max(order_by = heat_value, n = 1, with_ties = FALSE) %>%
  dplyr::ungroup()

gene_heat_df <- gene_heat_df %>%
  dplyr::mutate(
    label_text = ifelse(n_disgenet_genes > 0, as.character(n_disgenet_genes), ""),
    label_color = ifelse(
      heat_value >= stats::quantile(heat_value, 0.72, na.rm = TRUE),
      "black", "white"
    )
  )

gene_top_df <- gene_heat_df %>%
  dplyr::group_by(LV_label) %>%
  dplyr::summarise(
    total_relevant_genes = sum(n_disgenet_genes, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::mutate(LV_label = factor(LV_label, levels = gene_col_order))

gene_right_df <- gene_heat_df %>%
  dplyr::group_by(Disease) %>%
  dplyr::summarise(
    total_relevant_genes = sum(n_disgenet_genes, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::mutate(Disease = factor(Disease, levels = gene_row_order))

# ------------------------------------------------------------
# Gene plots
# ------------------------------------------------------------
p_gene_heat <- ggplot2::ggplot(
  gene_heat_df,
  ggplot2::aes(x = LV_label, y = Disease, fill = heat_value)
) +
  ggplot2::geom_tile(width = 0.98, height = 0.98, color = "white", linewidth = 0.45) +
  ggplot2::geom_text(
    ggplot2::aes(label = label_text, color = label_color),
    size = 2.7,
    fontface = "bold",
    show.legend = FALSE
  ) +
  ggplot2::geom_tile(
    data = gene_peak_df,
    width = 0.98, height = 0.98,
    fill = NA,
    color = "white",
    linewidth = 0.95
  ) +
  ggplot2::scale_color_identity() +
  viridis::scale_fill_viridis(
    option = GENE_VIRIDIS_OPTION,
    direction = 1,
    name = "Gene relevance\nscore",
    labels = scales::label_number(accuracy = 0.01)
  ) +
  ggplot2::scale_y_discrete(
    limits = gene_row_display,
    drop = FALSE,
    expand = ggplot2::expansion(add = c(0, 0))
  ) +
  ggplot2::labs(
    title = "a  Disease-relevant DisGeNET gene capture across latent variables",
    subtitle = "Each cell reports the number of disease-relevant genes supported within an LV across latent draws;\ncolor shows their summed DisGeNET relevance.",
    x = "Latent variable",
    y = "Disease",
    caption = "Numbers show the count of disease-relevant genes supported across multiple latent draws within each LV; heat intensity reflects DisGeNET relevance weighted by cross-draw support and gene count."
  ) +
  theme_journal() +
  ggplot2::theme(legend.position = "right")

p_gene_top <- ggplot2::ggplot(
  gene_top_df,
  ggplot2::aes(x = LV_label, y = total_relevant_genes)
) +
  ggplot2::geom_col(width = 0.72, fill = "#666666") +
  ggplot2::geom_text(
    ggplot2::aes(label = total_relevant_genes),
    vjust = -0.22,
    size = 2.8
  ) +
  ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.10))) +
  ggplot2::labs(
    title = "b  Total disease-relevant genes captured by latent variable",
    subtitle = "Summed across diseases.",
    x = NULL,
    y = "Matched genes"
  ) +
  theme_journal() +
  ggplot2::theme(
    axis.text.x = ggplot2::element_blank(),
    axis.ticks.x = ggplot2::element_blank()
  )

p_gene_right <- ggplot2::ggplot(
  gene_right_df,
  ggplot2::aes(x = total_relevant_genes, y = Disease)
) +
  ggplot2::geom_col(width = 0.90, fill = "#666666") +
  ggplot2::geom_text(
    ggplot2::aes(label = total_relevant_genes),
    hjust = -0.08,
    size = 2.7
  ) +
  ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = c(0, 0.10))) +
  ggplot2::scale_y_discrete(
    limits = gene_row_display,
    drop = FALSE,
    expand = ggplot2::expansion(add = c(0, 0))
  ) +
  ggplot2::coord_cartesian(clip = "off") +
  ggplot2::labs(
    title = "c  Total disease-relevant genes captured by disease",
    subtitle = "Summed across latent variables.",
    x = "Matched genes",
    y = NULL
  ) +
  theme_journal() +
  ggplot2::theme(
    axis.text.y = ggplot2::element_blank(),
    axis.ticks.y = ggplot2::element_blank()
  )

gene_block <- patchwork::wrap_plots(
  A = p_gene_top,
  B = patchwork::plot_spacer(),
  C = p_gene_heat,
  D = p_gene_right,
  design = "
AB
CD
",
  heights = c(0.82, 1.88),
  widths  = c(1.9, 1.05),
  guides = "collect"
)

# ============================================================
# 4. PATHWAY ANALYSIS
# ============================================================

pathway_files <- list.files(
  path = data_dir,
  pattern = pathway_file_pattern,
  full.names = TRUE,
  recursive = TRUE
)

if (length(pathway_files) == 0) {
  stop("No overlap_terms_by_library_topN_<DISEASE>_LD*_NS*_L*.csv files found recursively under ~/Documents")
}

extract_pathway_disease_from_filename <- function(path) {
  nm <- basename(path)
  m <- stringr::str_match(
    nm,
    "^overlap_terms_by_library_topN_([A-Za-z0-9]+)_LD[0-9]+_NS[0-9]+_L[0-9]+\\.csv$"
  )
  m[, 2]
}

extract_pathway_config_from_filename <- function(path) {
  nm <- basename(path)
  cfg <- stringr::str_extract(nm, "LD[0-9]+_NS[0-9]+_L[0-9]+")
  ifelse(is.na(cfg), "UNKNOWN_CFG", cfg)
}

normalize_pathway_name <- function(x) {
  x %>%
    as.character() %>%
    stringr::str_replace_all("_", " ") %>%
    stringr::str_replace_all("\\s+", " ") %>%
    stringr::str_squish()
}

build_display_pathway <- function(library, term) {
  lib_short <- dplyr::case_when(
    stringr::str_detect(tolower(library), "reactome") ~ "REAC",
    stringr::str_detect(tolower(library), "kegg") ~ "KEGG",
    stringr::str_detect(tolower(library), "^go") ~ "GO",
    TRUE ~ as.character(library)
  )
  paste0(lib_short, ": ", normalize_pathway_name(term))
}

make_pathway_axis_label <- function(x) {
  x <- x %>%
    stringr::str_replace("^REAC: ", "") %>%
    stringr::str_replace("^KEGG: ", "") %>%
    stringr::str_replace("^GO: ", "") %>%
    stringr::str_replace_all("\\(.*?\\)", "") %>%
    stringr::str_replace_all("[^A-Za-z0-9 ]", " ") %>%
    stringr::str_squish()
  
  x <- x %>%
    stringr::str_replace_all("\\bPositive Regulation Of\\b", "PosReg") %>%
    stringr::str_replace_all("\\bNegative Regulation Of\\b", "NegReg") %>%
    stringr::str_replace_all("\\bRegulation Of\\b", "Reg") %>%
    stringr::str_replace_all("\\bResponse To\\b", "Resp") %>%
    stringr::str_replace_all("\\bSignaling By\\b", "") %>%
    stringr::str_replace_all("\\bPathway\\b", "") %>%
    stringr::str_replace_all("\\bBiosynthetic Process\\b", "Biosynth") %>%
    stringr::str_replace_all("\\bMetabolic Process\\b", "Metab") %>%
    stringr::str_replace_all("\\bOrganization\\b", "Org") %>%
    stringr::str_replace_all("\\bActivation\\b", "Act") %>%
    stringr::str_replace_all("\\bTransport\\b", "Transp") %>%
    stringr::str_replace_all("\\bCholesterol\\b", "Chol") %>%
    stringr::str_replace_all("\\bMitochondrion\\b", "Mito") %>%
    stringr::str_replace_all("\\bLymphocyte\\b", "Lymph") %>%
    stringr::str_replace_all("\\bImmunity\\b", "Immune") %>%
    stringr::str_replace_all("\\bOxidative Stress\\b", "OxStress") %>%
    stringr::str_replace_all("\\bTyrosine Kinase\\b", "RTK") %>%
    stringr::str_squish()
  
  words <- unlist(strsplit(x, "\\s+"))
  if (length(words) == 0) return("Pathway")
  
  short <- paste0(substr(words, 1, pmin(nchar(words), 4)), collapse = "")
  short <- stringr::str_replace_all(short, "[^A-Za-z0-9]", "")
  stringr::str_trunc(short, width = 16, side = "right", ellipsis = "…")
}

pathway_file_diag <- tibble(pathway_file = pathway_files) %>%
  dplyr::mutate(
    Disease = extract_pathway_disease_from_filename(pathway_file),
    config = extract_pathway_config_from_filename(pathway_file),
    file = basename(pathway_file)
  )

readr::write_csv(
  pathway_file_diag,
  file.path(data_dir, "pathway_file_diagnostics_combined.csv")
)

pathway_raw <- purrr::map_dfr(pathway_files, function(f) {
  disease <- extract_pathway_disease_from_filename(f)
  config <- extract_pathway_config_from_filename(f)
  
  if (is.na(disease)) return(NULL)
  
  df <- tryCatch(
    readr::read_csv(f, show_col_types = FALSE),
    error = function(e) NULL
  )
  if (is.null(df) || nrow(df) == 0) return(NULL)
  
  required_cols <- c(
    "Library", "Term", "BestFDR", "MedianFDR", "MeanFDR",
    "BestScore", "MedianScore", "MeanScore",
    "NumSamplesTopN", "NumLVsTopN",
    "RepRateSamplesTopN", "RepRateLVsTopN"
  )
  
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) return(NULL)
  
  df %>%
    dplyr::transmute(
      Disease = disease,
      config = config,
      Library = as.character(Library),
      pathway = build_display_pathway(Library, Term),
      BestFDR = suppressWarnings(as.numeric(BestFDR)),
      MedianFDR = suppressWarnings(as.numeric(MedianFDR)),
      MeanFDR = suppressWarnings(as.numeric(MeanFDR)),
      BestScore = suppressWarnings(as.numeric(BestScore)),
      MedianScore = suppressWarnings(as.numeric(MedianScore)),
      MeanScore = suppressWarnings(as.numeric(MeanScore)),
      NumSamplesTopN = suppressWarnings(as.numeric(NumSamplesTopN)),
      NumLVsTopN = suppressWarnings(as.numeric(NumLVsTopN)),
      RepRateSamplesTopN = suppressWarnings(as.numeric(RepRateSamplesTopN)),
      RepRateLVsTopN = suppressWarnings(as.numeric(RepRateLVsTopN)),
      file = basename(f)
    ) %>%
    dplyr::filter(
      !is.na(pathway), pathway != "",
      !is.na(BestFDR), is.finite(BestFDR)
    )
})

if (nrow(pathway_raw) == 0) stop("No usable pathway rows found after reading the files.")

pathway_raw <- pathway_raw %>%
  dplyr::filter(BestFDR <= FDR_THRESHOLD)

if (nrow(pathway_raw) == 0) stop("No pathway entries passed the chosen FDR threshold.")

pathway_summary <- pathway_raw %>%
  dplyr::group_by(Disease, pathway) %>%
  dplyr::summarise(
    n_configs_hit = dplyr::n_distinct(config),
    best_fdr = min(BestFDR, na.rm = TRUE),
    median_fdr = median(BestFDR, na.rm = TRUE),
    mean_fdr = mean(BestFDR, na.rm = TRUE),
    best_score = max(BestScore, na.rm = TRUE),
    mean_score = mean(MeanScore, na.rm = TRUE),
    max_NumLVsTopN = max(NumLVsTopN, na.rm = TRUE),
    max_NumSamplesTopN = max(NumSamplesTopN, na.rm = TRUE),
    max_RepRateLVsTopN = max(RepRateLVsTopN, na.rm = TRUE),
    max_RepRateSamplesTopN = max(RepRateSamplesTopN, na.rm = TRUE),
    n_libraries = dplyr::n_distinct(Library),
    .groups = "drop"
  )

if (USE_NEGLOG10_FDR) {
  pathway_summary <- pathway_summary %>%
    dplyr::mutate(
      fdr_signal = -log10(pmax(best_fdr, 1e-300))
    )
} else {
  pathway_summary <- pathway_summary %>%
    dplyr::mutate(
      fdr_signal = 1 / pmax(best_fdr, 1e-300)
    )
}

pathway_summary <- pathway_summary %>%
  dplyr::mutate(
    support_rate = 0.5 * max_RepRateLVsTopN + 0.5 * max_RepRateSamplesTopN,
    recurrence_factor = log1p(n_configs_hit),
    intensity_raw = fdr_signal * recurrence_factor * pmax(support_rate, 1e-6)
  )

cap_value <- stats::quantile(
  pathway_summary$intensity_raw,
  probs = CAP_INTENSITY_AT_QUANTILE,
  na.rm = TRUE
)

pathway_summary <- pathway_summary %>%
  dplyr::mutate(
    intensity_capped = pmin(intensity_raw, cap_value)
  )

pathway_rank <- pathway_summary %>%
  dplyr::group_by(pathway) %>%
  dplyr::summarise(
    global_configs = sum(n_configs_hit, na.rm = TRUE),
    global_intensity = sum(intensity_capped, na.rm = TRUE),
    n_diseases = dplyr::n_distinct(Disease),
    .groups = "drop"
  ) %>%
  dplyr::arrange(
    dplyr::desc(global_intensity),
    dplyr::desc(global_configs),
    dplyr::desc(n_diseases),
    pathway
  )

top_pathways_multi <- pathway_rank %>%
  dplyr::filter(n_diseases >= MIN_DISEASE_SUPPORT_FOR_PATHWAY) %>%
  dplyr::slice_head(n = MAX_PATHWAYS_TO_SHOW) %>%
  dplyr::pull(pathway)

top_pathways <- if (length(top_pathways_multi) >= 8) {
  top_pathways_multi
} else {
  pathway_rank %>%
    dplyr::slice_head(n = MAX_PATHWAYS_TO_SHOW) %>%
    dplyr::pull(pathway)
}

path_heat_df <- pathway_summary %>%
  dplyr::filter(pathway %in% top_pathways)

if (nrow(path_heat_df) == 0) stop("No pathway data remained after pathway selection.")

all_path_diseases <- sort(unique(pathway_summary$Disease))
all_pathways <- unique(top_pathways)

path_heat_df <- tidyr::expand_grid(
  Disease = all_path_diseases,
  pathway = all_pathways
) %>%
  dplyr::left_join(path_heat_df, by = c("Disease", "pathway")) %>%
  dplyr::mutate(
    n_configs_hit = tidyr::replace_na(n_configs_hit, 0L),
    best_fdr = tidyr::replace_na(best_fdr, 1),
    median_fdr = tidyr::replace_na(median_fdr, 1),
    mean_fdr = tidyr::replace_na(mean_fdr, 1),
    best_score = tidyr::replace_na(best_score, 0),
    mean_score = tidyr::replace_na(mean_score, 0),
    max_NumLVsTopN = tidyr::replace_na(max_NumLVsTopN, 0),
    max_NumSamplesTopN = tidyr::replace_na(max_NumSamplesTopN, 0),
    max_RepRateLVsTopN = tidyr::replace_na(max_RepRateLVsTopN, 0),
    max_RepRateSamplesTopN = tidyr::replace_na(max_RepRateSamplesTopN, 0),
    n_libraries = tidyr::replace_na(n_libraries, 0L),
    fdr_signal = tidyr::replace_na(fdr_signal, 0),
    support_rate = tidyr::replace_na(support_rate, 0),
    recurrence_factor = tidyr::replace_na(recurrence_factor, 0),
    intensity_raw = tidyr::replace_na(intensity_raw, 0),
    intensity_capped = tidyr::replace_na(intensity_capped, 0)
  )

# ------------------------------------------------------------
# Pathway ordering
# ------------------------------------------------------------
path_heat_mat <- path_heat_df %>%
  dplyr::select(Disease, pathway, intensity_capped) %>%
  tidyr::pivot_wider(names_from = pathway, values_from = intensity_capped) %>%
  dplyr::arrange(Disease)

path_row_names <- path_heat_mat$Disease
path_heat_mat_num <- path_heat_mat %>%
  dplyr::select(-Disease) %>%
  as.matrix()

rownames(path_heat_mat_num) <- path_row_names

if (ncol(path_heat_mat_num) > 1) {
  path_col_hc <- stats::hclust(stats::dist(t(path_heat_mat_num)), method = "ward.D2")
  path_col_order <- colnames(path_heat_mat_num)[path_col_hc$order]
  
  path_col_avg <- colMeans(path_heat_mat_num[, path_col_order, drop = FALSE], na.rm = TRUE)
  if (mean(path_col_avg[1:min(3, length(path_col_avg))], na.rm = TRUE) >
      mean(tail(path_col_avg, min(3, length(path_col_avg))), na.rm = TRUE)) {
    path_col_order <- path_col_order
  }
} else {
  path_col_order <- colnames(path_heat_mat_num)
}

path_row_score_df <- path_heat_df %>%
  dplyr::group_by(Disease) %>%
  dplyr::summarise(
    avg_heat = mean(intensity_capped, na.rm = TRUE),
    max_heat = max(intensity_capped, na.rm = TRUE),
    total_heat = sum(intensity_capped, na.rm = TRUE),
    total_hits = sum(n_configs_hit, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(avg_heat, max_heat, total_heat, total_hits)

path_row_order <- path_row_score_df$Disease
path_row_display <- path_row_order

path_heat_df <- path_heat_df %>%
  dplyr::mutate(
    Disease = factor(Disease, levels = path_row_order),
    pathway = factor(pathway, levels = path_col_order)
  )

path_peak_df <- path_heat_df %>%
  dplyr::group_by(Disease) %>%
  dplyr::slice_max(order_by = intensity_capped, n = 1, with_ties = FALSE) %>%
  dplyr::ungroup()

path_top_df <- path_heat_df %>%
  dplyr::group_by(pathway) %>%
  dplyr::summarise(
    total_pathway_hits = sum(n_configs_hit, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::mutate(pathway = factor(pathway, levels = path_col_order))

path_right_df <- path_heat_df %>%
  dplyr::group_by(Disease) %>%
  dplyr::summarise(
    total_pathway_hits = sum(n_configs_hit, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::mutate(Disease = factor(Disease, levels = path_row_order))

pathway_label_map <- tibble(
  pathway = path_col_order
) %>%
  dplyr::mutate(
    pathway_display = vapply(pathway, make_pathway_axis_label, character(1)),
    pathway_display = make.unique(pathway_display, sep = "_"),
    pathway_key = pathway
  )

path_heat_df <- path_heat_df %>%
  dplyr::left_join(pathway_label_map, by = "pathway")

path_top_df <- path_top_df %>%
  dplyr::left_join(pathway_label_map, by = "pathway")

path_peak_df <- path_peak_df %>%
  dplyr::left_join(pathway_label_map, by = "pathway")

readr::write_csv(
  pathway_label_map,
  file.path(data_dir, "pathway_label_key_combined_figure.csv")
)

p_path_heat <- ggplot2::ggplot(
  path_heat_df,
  ggplot2::aes(x = pathway_display, y = Disease, fill = intensity_capped)
) +
  ggplot2::geom_tile(width = 0.98, height = 0.98, color = "white", linewidth = 0.55) +
  ggplot2::geom_tile(
    data = path_peak_df,
    width = 0.98, height = 0.98,
    fill = NA,
    color = "white",
    linewidth = 1.0
  ) +
  viridis::scale_fill_viridis(
    option = PATHWAY_VIRIDIS_OPTION,
    direction = 1,
    name = "Pathway relevance\nscore",
    labels = scales::label_number(accuracy = 0.01)
  ) +
  ggplot2::scale_y_discrete(
    limits = path_row_display,
    drop = FALSE,
    expand = ggplot2::expansion(add = c(0, 0))
  ) +
  ggplot2::labs(
    title = "d  Disease-relevant pathway capture across diseases",
    subtitle = "Color reflects pathway-relevance intensity across disease–pathway pairs.",
    x = "Pathway",
    y = "Disease",
    caption = "Full pathway names corresponding to the abbreviated x-axis labels are reported in the exported pathway key table."
  ) +
  theme_journal() +
  ggplot2::theme(
    axis.text.x = ggplot2::element_text(angle = 55, hjust = 1, vjust = 1, size = 8.2),
    legend.position = "right"
  )

p_path_top <- ggplot2::ggplot(
  path_top_df,
  ggplot2::aes(x = pathway_display, y = total_pathway_hits)
) +
  ggplot2::geom_col(width = 0.72, fill = "#666666") +
  ggplot2::geom_text(
    ggplot2::aes(label = total_pathway_hits),
    vjust = -0.22,
    size = 2.7
  ) +
  ggplot2::scale_y_continuous(
    expand = ggplot2::expansion(mult = c(0, 0.10)),
    breaks = scales::pretty_breaks(n = 4)
  ) +
  ggplot2::labs(
    title = "e  Total disease-relevant pathways captured by pathway",
    subtitle = "Summed across diseases.",
    x = NULL,
    y = "Configuration hits"
  ) +
  theme_journal() +
  ggplot2::theme(
    axis.text.x = ggplot2::element_blank(),
    axis.ticks.x = ggplot2::element_blank()
  )

p_path_right <- ggplot2::ggplot(
  path_right_df,
  ggplot2::aes(x = total_pathway_hits, y = Disease)
) +
  ggplot2::geom_col(width = 0.90, fill = "#666666") +
  ggplot2::geom_text(
    ggplot2::aes(label = total_pathway_hits),
    hjust = -0.08,
    size = 2.7
  ) +
  ggplot2::scale_x_continuous(
    expand = ggplot2::expansion(mult = c(0, 0.10)),
    breaks = scales::pretty_breaks(n = 4)
  ) +
  ggplot2::scale_y_discrete(
    limits = path_row_display,
    drop = FALSE,
    expand = ggplot2::expansion(add = c(0, 0))
  ) +
  ggplot2::coord_cartesian(clip = "off") +
  ggplot2::labs(
    title = "f  Total disease-relevant pathways captured by disease",
    subtitle = "Summed across pathways.",
    x = "Configuration hits",
    y = NULL
  ) +
  theme_journal() +
  ggplot2::theme(
    axis.text.y = ggplot2::element_blank(),
    axis.ticks.y = ggplot2::element_blank()
  )

path_block <- patchwork::wrap_plots(
  A = p_path_top,
  B = patchwork::plot_spacer(),
  C = p_path_heat,
  D = p_path_right,
  design = "
AB
CD
",
  heights = c(0.82, 1.88),
  widths  = c(1.9, 1.05),
  guides = "collect"
)

# ============================================================
# 5. FINAL CLEAN FIGURE
# ============================================================
final_plot <- patchwork::wrap_plots(
  gene_block,
  path_block,
  ncol = 1,
  heights = c(1, 1),
  guides = "collect"
) &
  ggplot2::theme(
    legend.position = "right",
    plot.background = ggplot2::element_rect(fill = "white", colour = NA),
    plot.margin = ggplot2::margin(1, 1, 1, 1)
  )

# ============================================================
# 6. BIOLOGICAL ANNOTATION HELPERS
# ============================================================

make_callout_grob <- function(
    title,
    body,
    fill = "#EEF4FF",
    border = "#5679C1",
    index = NULL,
    title_wrap = 28,
    body_wrap = 34,
    title_cex = 0.88,
    body_cex = 0.72
) {
  title_wrapped <- stringr::str_wrap(title, width = title_wrap)
  body_wrapped  <- stringr::str_wrap(body,  width = body_wrap)
  
  grob_list <- list(
    grid::roundrectGrob(
      x = 0.5, y = 0.5,
      width = 0.98, height = 0.98,
      r = grid::unit(0.05, "snpc"),
      gp = grid::gpar(fill = fill, col = border, lwd = 1.4)
    )
  )
  
  if (!is.null(index)) {
    grob_list <- append(
      grob_list,
      list(
        grid::circleGrob(
          x = grid::unit(0.10, "npc"),
          y = grid::unit(0.88, "npc"),
          r = grid::unit(0.06, "npc"),
          gp = grid::gpar(fill = border, col = border, lwd = 1)
        ),
        grid::textGrob(
          label = as.character(index),
          x = grid::unit(0.10, "npc"),
          y = grid::unit(0.88, "npc"),
          gp = grid::gpar(col = "white", fontsize = 10, fontface = "bold")
        )
      )
    )
    title_x <- 0.20
  } else {
    title_x <- 0.06
  }
  
  grob_list <- append(
    grob_list,
    list(
      grid::textGrob(
        label = title_wrapped,
        x = grid::unit(title_x, "npc"),
        y = grid::unit(0.88, "npc"),
        just = c("left", "top"),
        gp = grid::gpar(
          col = "black",
          fontsize = 10.8,
          fontface = "bold",
          lineheight = 1.0
        )
      ),
      grid::textGrob(
        label = body_wrapped,
        x = grid::unit(0.06, "npc"),
        y = grid::unit(0.68, "npc"),
        just = c("left", "top"),
        gp = grid::gpar(
          col = "#222222",
          fontsize = 8.7,
          fontface = "plain",
          lineheight = 1.08
        )
      )
    )
  )
  
  grid::grobTree(children = do.call(grid::gList, grob_list))
}

# ---- Callout palette
COL_LATENT_FILL  <- "#EAF2FF"
COL_LATENT_LINE  <- "#4F77BE"

COL_GENE_FILL    <- "#FFF0E6"
COL_GENE_LINE    <- "#D9822B"

COL_PATH_FILL    <- "#FFF7D6"
COL_PATH_LINE    <- "#B79C24"

# ============================================================
# 7. BIOLOGICAL CALLOUT TEXT
# ============================================================

callout_1_title <- "Distributed latent signal"
callout_1_body  <- "Similar gene totals across latent variables indicate that gVAE distributes disease-relevant signal across multiple latent dimensions, rather than relying on a single dominant LV."

callout_2_title <- "Robust systemic disease capture"
callout_2_body  <- "Strong gene relevance in HT, CAD, T2D, RA, BMI, HDL, and LDL supports recovery of shared systemic cardiometabolic and inflammatory biology across the latent space."

callout_3_title <- "Narrower or context-dependent signatures"
callout_3_body  <- "Lower gene overlap in ASD, COL, LUN, and PRC may reflect more specific, heterogeneous, or context-dependent molecular signatures rather than a lack of meaningful signal."

callout_4_title <- "ALZ is stronger at the pathway level"
callout_4_body  <- "Alzheimer-related signal is more prominent in pathway recurrence than in direct disease-gene overlap, suggesting that ALZ structure emerges more coherently at the systems level."

callout_5_title <- "Pathways persist beyond gene overlap"
callout_5_body  <- "COL and LUN retain pathway recurrence despite relatively modest gene-level overlap, consistent with convergent biological programs being recovered even when direct gene matches are sparse."

callout_6_title <- "Shared pathway backbone"
callout_6_body  <- "Repeated bright pathway bands across diseases suggest that gVAE captures a shared higher-order biological backbone alongside disease-specific effects."

# ============================================================
# 8. CLEAN + ANNOTATED FIGURES
# ============================================================

# ---- Clean figure
final_plot_clean <- final_plot

# ---- Annotated figure
# The clean plot is drawn slightly smaller to make room for callout boxes.
final_plot_annotated <- cowplot::ggdraw() +
  cowplot::draw_plot(final_plot_clean, x = 0.12, y = 0.045, width = 0.76, height = 0.87) +
  
  # ----------------------------------------------------------
# Top callout (latent-space interpretation)
# ----------------------------------------------------------
cowplot::draw_grob(
  make_callout_grob(
    title = callout_1_title,
    body  = callout_1_body,
    fill = COL_LATENT_FILL,
    border = COL_LATENT_LINE,
    index = 1,
    title_wrap = 36,
    body_wrap = 70
  ),
  x = 0.24, y = 0.915, width = 0.42, height = 0.075
) +
  
  ggplot2::annotate(
    "curve",
    x = 0.45, y = 0.915,
    xend = 0.42, yend = 0.855,
    curvature = -0.20,
    colour = COL_LATENT_LINE,
    linewidth = 0.7,
    arrow = grid::arrow(length = grid::unit(0.012, "npc"), type = "closed")
  ) +
  
  # ----------------------------------------------------------
# Left gene-side callouts
# ----------------------------------------------------------
cowplot::draw_grob(
  make_callout_grob(
    title = callout_2_title,
    body  = callout_2_body,
    fill = COL_GENE_FILL,
    border = COL_GENE_LINE,
    index = 2,
    title_wrap = 22,
    body_wrap = 25
  ),
  x = 0.005, y = 0.63, width = 0.11, height = 0.18
) +
  
  ggplot2::annotate(
    "curve",
    x = 0.115, y = 0.72,
    xend = 0.34, yend = 0.70,
    curvature = -0.18,
    colour = COL_GENE_LINE,
    linewidth = 0.7,
    arrow = grid::arrow(length = grid::unit(0.012, "npc"), type = "closed")
  ) +
  
  cowplot::draw_grob(
    make_callout_grob(
      title = callout_3_title,
      body  = callout_3_body,
      fill = COL_GENE_FILL,
      border = COL_GENE_LINE,
      index = 3,
      title_wrap = 22,
      body_wrap = 25
    ),
    x = 0.005, y = 0.43, width = 0.11, height = 0.17
  ) +
  
  ggplot2::annotate(
    "curve",
    x = 0.115, y = 0.50,
    xend = 0.30, yend = 0.56,
    curvature = 0.16,
    colour = COL_GENE_LINE,
    linewidth = 0.7,
    arrow = grid::arrow(length = grid::unit(0.012, "npc"), type = "closed")
  ) +
  
  # ----------------------------------------------------------
# Right pathway-side callouts
# ----------------------------------------------------------
cowplot::draw_grob(
  make_callout_grob(
    title = callout_6_title,
    body  = callout_6_body,
    fill = COL_PATH_FILL,
    border = COL_PATH_LINE,
    index = 6,
    title_wrap = 22,
    body_wrap = 25
  ),
  x = 0.885, y = 0.32, width = 0.11, height = 0.16
) +
  
  ggplot2::annotate(
    "curve",
    x = 0.885, y = 0.39,
    xend = 0.47, yend = 0.28,
    curvature = 0.14,
    colour = COL_PATH_LINE,
    linewidth = 0.7,
    arrow = grid::arrow(length = grid::unit(0.012, "npc"), type = "closed")
  ) +
  
  cowplot::draw_grob(
    make_callout_grob(
      title = callout_4_title,
      body  = callout_4_body,
      fill = COL_PATH_FILL,
      border = COL_PATH_LINE,
      index = 4,
      title_wrap = 22,
      body_wrap = 25
    ),
    x = 0.885, y = 0.15, width = 0.11, height = 0.16
  ) +
  
  ggplot2::annotate(
    "curve",
    x = 0.885, y = 0.23,
    xend = 0.74, yend = 0.34,
    curvature = -0.12,
    colour = COL_PATH_LINE,
    linewidth = 0.7,
    arrow = grid::arrow(length = grid::unit(0.012, "npc"), type = "closed")
  ) +
  
  cowplot::draw_grob(
    make_callout_grob(
      title = callout_5_title,
      body  = callout_5_body,
      fill = COL_PATH_FILL,
      border = COL_PATH_LINE,
      index = 5,
      title_wrap = 22,
      body_wrap = 25
    ),
    x = 0.885, y = 0.005, width = 0.11, height = 0.14
  ) +
  
  ggplot2::annotate(
    "curve",
    x = 0.885, y = 0.08,
    xend = 0.73, yend = 0.23,
    curvature = -0.12,
    colour = COL_PATH_LINE,
    linewidth = 0.7,
    arrow = grid::arrow(length = grid::unit(0.012, "npc"), type = "closed")
  )

# ------------------------------------------------------------
# 9. DISPLAY
# ------------------------------------------------------------
print(final_plot_clean)
print(final_plot_annotated)

# ------------------------------------------------------------
# 10. SAVE
# ------------------------------------------------------------
pdf_device_to_use <- "pdf"

# ---- Clean figure
ggplot2::ggsave(
  filename = paste0(out_prefix_clean, ".pdf"),
  plot = final_plot_clean,
  width = 15.5,
  height = 15.6,
  units = "in",
  dpi = 600,
  device = pdf_device_to_use,
  bg = "white"
)

ggplot2::ggsave(
  filename = paste0(out_prefix_clean, ".png"),
  plot = final_plot_clean,
  width = 15.5,
  height = 15.6,
  units = "in",
  dpi = 600,
  bg = "white"
)

ggplot2::ggsave(
  filename = paste0(out_prefix_clean, ".tiff"),
  plot = final_plot_clean,
  width = 15.5,
  height = 15.6,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)

# ---- Annotated figure
ggplot2::ggsave(
  filename = paste0(out_prefix_annotated, ".pdf"),
  plot = final_plot_annotated,
  width = 18.5,
  height = 16.4,
  units = "in",
  dpi = 600,
  device = pdf_device_to_use,
  bg = "white"
)

ggplot2::ggsave(
  filename = paste0(out_prefix_annotated, ".png"),
  plot = final_plot_annotated,
  width = 18.5,
  height = 16.4,
  units = "in",
  dpi = 600,
  bg = "white"
)

ggplot2::ggsave(
  filename = paste0(out_prefix_annotated, ".tiff"),
  plot = final_plot_annotated,
  width = 18.5,
  height = 16.4,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)

# ------------------------------------------------------------
# 11. EXPORT CHECK TABLES
# ------------------------------------------------------------
readr::write_csv(
  gene_heat_df %>%
    dplyr::select(
      Disease, Config, LV_label, n_disgenet_genes,
      total_disgenet_genes_for_disease,
      capture_fraction, weighted_fraction, heat_value,
      mean_support_fraction, total_gene_count
    ),
  file.path(data_dir, "combined_figure_gene_heatmap_summary.csv")
)

readr::write_csv(
  path_heat_df %>%
    dplyr::select(
      Disease, pathway, pathway_display, n_configs_hit,
      best_fdr, median_fdr, mean_fdr,
      best_score, mean_score,
      max_NumLVsTopN, max_NumSamplesTopN,
      max_RepRateLVsTopN, max_RepRateSamplesTopN,
      n_libraries, intensity_raw, intensity_capped
    ),
  file.path(data_dir, "combined_figure_pathway_heatmap_summary.csv")
)
