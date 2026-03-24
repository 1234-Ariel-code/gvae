#!/usr/bin/env Rscript

# ============================================================
# Flat latent-folder version for ARC/local use
# ============================================================

# ------------------------------------------------------------
# REQUIRED R PACKAGES
# ------------------------------------------------------------
required_pkgs <- c(
  "readr",
  "dplyr",
  "tidyr",
  "ggplot2",
  "stringr",
  "purrr",
  "tibble",
  "patchwork",
  "scales",
  "glue"
)

missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    paste0(
      "Missing required R packages: ",
      paste(missing_pkgs, collapse = ", "),
      ".\nInstall them into the active R user library before submitting the SLURM job."
    )
  )
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
  library(purrr)
  library(tibble)
  library(patchwork)
  library(scales)
  library(glue)
})

# ------------------------------------------------------------
# 0. ARGUMENTS
# ------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

get_arg_value <- function(flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default)
  if (idx == length(args)) return(default)
  args[idx + 1]
}

MODE <- get_arg_value("--mode", "cluster")

latent_root_arg   <- get_arg_value("--latent_root", NULL)
gwas_root_arg     <- get_arg_value("--gwas_root", NULL)
bim_root_arg      <- get_arg_value("--bim_root", NULL)
cs2g_file_arg     <- get_arg_value("--cs2g_file", NULL)
disgenet_file_arg <- get_arg_value("--disgenet_file", NULL)
drug_file_arg     <- get_arg_value("--drug_target_file", NULL)
out_dir_arg       <- get_arg_value("--out_dir", NULL)

# ------------------------------------------------------------
# 1. PATH CONFIGURATION
# ------------------------------------------------------------
if (MODE == "cluster") {
  latent_root_default   <- "/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/latent"
  gwas_root_default     <- "/work/long_lab/for_Ariel/gwas_results"
  bim_root_default      <- "/work/long_lab/for_Ariel/files"
  cs2g_file_default     <- "/work/long_lab/for_Ariel/files/combined_cS2G.tsv"
  disgenet_file_default <- "/work/long_lab/for_Ariel/files/consolidated.tsv"
  drug_file_default     <- "/work/long_lab/for_Ariel/files/drug_target_gene_sets.tsv"
  out_dir_default       <- "/work/long_lab/Ariel_Kemogne/Representation_learning/AJHG/_dynamic_gwas_comparison"
} else {
  latent_root_default   <- "~/Documents"
  gwas_root_default     <- "~/Documents"
  bim_root_default      <- "~/Documents"
  cs2g_file_default     <- "~/Documents/combined_cS2G.tsv"
  disgenet_file_default <- "~/Documents/consolidated.tsv"
  drug_file_default     <- "~/Documents/drug_target_gene_sets.tsv"
  out_dir_default       <- "~/Documents/dynamic_gwas_comparison"
}

latent_root      <- path.expand(ifelse(is.null(latent_root_arg),   latent_root_default,   latent_root_arg))
gwas_root        <- path.expand(ifelse(is.null(gwas_root_arg),     gwas_root_default,     gwas_root_arg))
bim_root         <- path.expand(ifelse(is.null(bim_root_arg),      bim_root_default,      bim_root_arg))
cs2g_file        <- path.expand(ifelse(is.null(cs2g_file_arg),     cs2g_file_default,     cs2g_file_arg))
disgenet_file    <- path.expand(ifelse(is.null(disgenet_file_arg), disgenet_file_default, disgenet_file_arg))
drug_target_file <- path.expand(ifelse(is.null(drug_file_arg),     drug_file_default,     drug_file_arg))
out_dir          <- path.expand(ifelse(is.null(out_dir_arg),       out_dir_default,       out_dir_arg))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("MODE: ", MODE)
message("latent_root   = ", latent_root)
message("gwas_root     = ", gwas_root)
message("bim_root      = ", bim_root)
message("cs2g_file     = ", cs2g_file)
message("disgenet_file = ", disgenet_file)
message("drug_target_file = ", drug_target_file)
message("out_dir       = ", out_dir)

# ------------------------------------------------------------
# 2. USER SETTINGS
# ------------------------------------------------------------
MAX_K_PER_LV <- 10
USE_COMBINED_OBJECTIVE <- TRUE
W_DISGENET <- 1.0
W_TARGET   <- 0.5
USE_DISGENET_THERAPEUTIC_PROXY <- TRUE
MIN_FINAL_UNION_SIZE <- 3

out_prefix_main <- file.path(out_dir, "Figure_DynamicRepresentation_vs_GWAS")
out_prefix_supp <- file.path(out_dir, "Figure_DynamicPerLV_vs_GWAS")

# ------------------------------------------------------------
# 3. DISEASE -> DisGeNET TERM MAP
# ------------------------------------------------------------
disease_map <- tribble(
  ~Disease, ~DisGeNET_term,
  "ALZ", "Alzheimer's disease",
  "ASD", "autistic disorder",
  "BD",  "bipolar disorder",
  "BMI", "obesity",
  "BRC", "breast cancer",
  "CAD", "coronary artery disease",
  "CD",  "ulcerative colitis",
  "COL", "colon cancer",
  "EOS", "Barrett's esophagus",
  "HDL", "metabolic syndrome X",
  "HGT", "osteoporosis",
  "HT",  "hypertension",
  "LDL", "metabolic syndrome X",
  "LUN", "lung cancer",
  "PRC", "prostate cancer",
  "RA",  "rheumatoid arthritis",
  "T1D", "type 1 diabetes mellitus",
  "T2D", "type 2 diabetes mellitus"
)

# ------------------------------------------------------------
# 4. HELPERS
# ------------------------------------------------------------
extract_disease_from_latent <- function(path) {
  nm <- basename(path)
  sub("^latent_gene_table\\.ALL_SAMPLES\\.long_([A-Za-z0-9]+)_LD.*\\.csv$", "\\1", nm)
}

extract_config_from_latent <- function(path) {
  nm <- basename(path)
  cfg <- stringr::str_extract(nm, "LD[0-9]+_NS[0-9]+_L[0-9]+")
  cfg
}

extract_disease_from_gwas <- function(path) {
  nm <- basename(path)
  sub("^([A-Za-z0-9]+)_gwas\\.(assoc|qassoc)$", "\\1", nm)
}

read_latent <- function(path) {
  disease <- extract_disease_from_latent(path)
  config  <- extract_config_from_latent(path)

  read_csv(path, show_col_types = FALSE) %>%
    transmute(
      Disease = disease,
      Config = config,
      Latent_Dim = as.character(Latent_Dim),
      LV_num = suppressWarnings(as.integer(stringr::str_extract(Latent_Dim, "\\d+"))),
      LV = paste0("LV", LV_num + 1),
      SNP = as.character(SNP_ID),
      GENE = toupper(trimws(as.character(GENE))),
      Sample = as.character(Sample),
      latent_file = path
    ) %>%
    filter(!is.na(LV_num), !is.na(SNP), SNP != "", !is.na(GENE), GENE != "") %>%
    distinct(Disease, Config, LV, SNP, GENE, Sample, .keep_all = TRUE)
}

read_gwas_robust <- function(path) {
  df <- read.table(
    path,
    header = TRUE,
    stringsAsFactors = FALSE,
    sep = "",
    check.names = FALSE,
    fill = TRUE,
    comment.char = ""
  ) %>%
    as_tibble()

  names(df) <- trimws(names(df))

  if (!"SNP" %in% names(df)) stop("GWAS file missing SNP column: ", basename(path))
  if (!"P" %in% names(df)) stop("GWAS file missing P column: ", basename(path))

  if (!"BP" %in% names(df)) {
    df <- df %>%
      mutate(BP = suppressWarnings(as.numeric(stringr::str_extract(SNP, "(?<=chr[0-9XYM]+_)[0-9]+"))))
  }

  if (!"CHR" %in% names(df)) {
    df <- df %>%
      mutate(CHR = stringr::str_extract(SNP, "(?<=chr)[0-9XYM]+"))
  }

  df %>%
    mutate(
      SNP = as.character(SNP),
      CHR = as.character(CHR),
      BP  = suppressWarnings(as.numeric(BP)),
      P   = suppressWarnings(as.numeric(P))
    )
}

load_bim_lookup_for_disease <- function(bim_root, disease) {
  bim_path <- file.path(bim_root, paste0(disease, ".bim"))
  if (!file.exists(bim_path)) return(NULL)

  read.table(bim_path, header = FALSE, stringsAsFactors = FALSE) %>%
    as_tibble() %>%
    transmute(
      CHR = as.character(V1),
      RSID = as.character(V2),
      BP = as.numeric(V4)
    ) %>%
    filter(!is.na(CHR), !is.na(BP), !is.na(RSID), RSID != "") %>%
    distinct(CHR, BP, .keep_all = TRUE)
}

resolve_gwas_snp_to_rsid <- function(gwas_df, bim_lookup = NULL) {
  out <- gwas_df %>%
    mutate(
      SNP = as.character(SNP),
      rsid = ifelse(grepl("^rs", SNP, ignore.case = TRUE), SNP, NA_character_)
    )

  if (!is.null(bim_lookup)) {
    need_map <- out %>% filter(is.na(rsid), !is.na(CHR), !is.na(BP))
    if (nrow(need_map) > 0) {
      mapped <- need_map %>%
        left_join(bim_lookup, by = c("CHR", "BP")) %>%
        mutate(rsid = RSID) %>%
        select(-RSID)

      out <- out %>%
        filter(!is.na(rsid)) %>%
        bind_rows(mapped)
    }
  }

  out %>%
    mutate(rsid = ifelse(grepl("^rs", rsid, ignore.case = TRUE), rsid, NA_character_))
}

score_gene_set <- function(genes, gene_set_tbl, disease) {
  if (length(genes) == 0) return(0L)
  ref <- gene_set_tbl %>%
    filter(Disease == disease) %>%
    pull(GENE) %>%
    unique()
  sum(unique(genes) %in% ref)
}

evaluate_candidate <- function(current_snps, candidate_snp, gwas_ranked_snps, disease,
                               cs2g_tbl, disgenet_tbl, drug_tbl) {
  new_set <- unique(c(current_snps, candidate_snp))
  k <- length(new_set)

  gwas_set <- gwas_ranked_snps[seq_len(min(k, length(gwas_ranked_snps)))]
  if (length(gwas_set) < k) return(NULL)

  lv_genes <- cs2g_tbl %>%
    filter(SNP %in% new_set) %>%
    pull(GENE) %>%
    unique()

  gwas_genes <- cs2g_tbl %>%
    filter(SNP %in% gwas_set) %>%
    pull(GENE) %>%
    unique()

  lv_dis   <- score_gene_set(lv_genes, disgenet_tbl, disease)
  gwas_dis <- score_gene_set(gwas_genes, disgenet_tbl, disease)

  lv_drug   <- score_gene_set(lv_genes, drug_tbl, disease)
  gwas_drug <- score_gene_set(gwas_genes, drug_tbl, disease)

  delta_dis  <- lv_dis - gwas_dis
  delta_drug <- lv_drug - gwas_drug

  objective <- if (USE_COMBINED_OBJECTIVE) {
    W_DISGENET * delta_dis + W_TARGET * delta_drug
  } else {
    delta_dis
  }

  tibble(
    SNP = candidate_snp,
    k = k,
    LV_DisGeNET = lv_dis,
    GWAS_DisGeNET = gwas_dis,
    Delta_DisGeNET = delta_dis,
    LV_Drug = lv_drug,
    GWAS_Drug = gwas_drug,
    Delta_Drug = delta_drug,
    objective = objective
  )
}

greedy_select_lv <- function(candidate_snps, gwas_ranked_snps, disease, lv,
                             cs2g_tbl, disgenet_tbl, drug_tbl, max_k) {
  selected <- character(0)
  path <- list()

  max_k <- min(max_k, length(candidate_snps), length(gwas_ranked_snps))
  if (max_k < 1) return(NULL)

  remaining <- candidate_snps

  for (step in seq_len(max_k)) {
    cand_eval <- map_dfr(
      remaining,
      ~ evaluate_candidate(
        current_snps = selected,
        candidate_snp = .x,
        gwas_ranked_snps = gwas_ranked_snps,
        disease = disease,
        cs2g_tbl = cs2g_tbl,
        disgenet_tbl = disgenet_tbl,
        drug_tbl = drug_tbl
      )
    )

    if (nrow(cand_eval) == 0) break

    best <- cand_eval %>%
      arrange(desc(objective), desc(Delta_DisGeNET), desc(Delta_Drug), SNP) %>%
      slice(1)

    selected <- c(selected, best$SNP)
    remaining <- setdiff(remaining, best$SNP)

    path[[step]] <- best %>%
      mutate(
        Disease = disease,
        LV = lv,
        step = step,
        selected_SNP = best$SNP
      )

    if (length(remaining) == 0) break
  }

  bind_rows(path)
}

find_latent_files <- function(latent_root) {
  if (!dir.exists(latent_root)) {
    stop("latent_root directory does not exist: ", latent_root)
  }

  latent_files <- list.files(
    path = latent_root,
    pattern = "^latent_gene_table\\.ALL_SAMPLES\\.long_.*\\.csv$",
    full.names = TRUE,
    recursive = TRUE
  )

  latent_files <- unique(latent_files)

  message("Number of latent files found: ", length(latent_files))
  if (length(latent_files) > 0) {
    message("Example latent files:")
    print(utils::head(latent_files, 20))
  }

  latent_files
}

# ------------------------------------------------------------
# 5. CHECK INPUT FILES
# ------------------------------------------------------------
if (!file.exists(cs2g_file)) stop("combined_cS2G.tsv not found: ", cs2g_file)
if (!file.exists(disgenet_file)) stop("consolidated.tsv not found: ", disgenet_file)

# ------------------------------------------------------------
# 6. LOAD cS2G
# ------------------------------------------------------------
cs2g <- read_tsv(cs2g_file, show_col_types = FALSE) %>%
  transmute(
    SNP = as.character(SNP),
    GENE = toupper(trimws(as.character(GENE)))
  ) %>%
  filter(!is.na(SNP), !is.na(GENE), SNP != "", GENE != "") %>%
  distinct()

# ------------------------------------------------------------
# 7. LOAD DisGeNET
# ------------------------------------------------------------
disgenet_raw <- read_tsv(disgenet_file, show_col_types = FALSE) %>%
  transmute(
    doid_name = trimws(as.character(doid_name)),
    geneSymbol = toupper(trimws(as.character(geneSymbol))),
    associationType = as.character(associationType)
  ) %>%
  filter(!is.na(doid_name), !is.na(geneSymbol), geneSymbol != "") %>%
  distinct()

disgenet_sets <- disease_map %>%
  inner_join(
    disgenet_raw %>% distinct(doid_name, geneSymbol),
    by = c("DisGeNET_term" = "doid_name"),
    relationship = "many-to-many"
  ) %>%
  transmute(Disease, GENE = geneSymbol) %>%
  distinct()

therapeutic_proxy_sets <- disease_map %>%
  inner_join(
    disgenet_raw %>%
      filter(grepl("Therapeutic", associationType, ignore.case = TRUE)) %>%
      distinct(doid_name, geneSymbol),
    by = c("DisGeNET_term" = "doid_name"),
    relationship = "many-to-many"
  ) %>%
  transmute(Disease, GENE = geneSymbol) %>%
  distinct()

# ------------------------------------------------------------
# 8. LOAD DRUG TARGET SETS
# ------------------------------------------------------------
if (file.exists(drug_target_file)) {
  drug_sets <- read_tsv(drug_target_file, show_col_types = FALSE) %>%
    transmute(
      Disease = as.character(Disease),
      GENE = toupper(trimws(as.character(GENE)))
    ) %>%
    filter(!is.na(Disease), !is.na(GENE), GENE != "") %>%
    distinct()
  drug_label <- "Drug-target"
} else if (USE_DISGENET_THERAPEUTIC_PROXY) {
  drug_sets <- therapeutic_proxy_sets
  drug_label <- "Therapeutic target"
} else {
  stop("No drug_target_gene_sets.tsv found and therapeutic proxy disabled.")
}

# ------------------------------------------------------------
# 9. LOAD LATENT + GWAS FILES
# ------------------------------------------------------------
latent_files <- find_latent_files(latent_root)
gwas_files <- list.files(gwas_root, pattern = "(_gwas\\.(assoc|qassoc))$", full.names = TRUE)

if (length(latent_files) == 0) stop("No latent_gene_table files found under: ", latent_root)
if (length(gwas_files) == 0) stop("No *_gwas.assoc or *_gwas.qassoc files found under: ", gwas_root)

message("Found ", length(latent_files), " latent files")
message("Found ", length(gwas_files), " GWAS files")

latent_all <- map_dfr(latent_files, read_latent)

gwas_tbl <- tibble(
  Disease = map_chr(gwas_files, extract_disease_from_gwas),
  gwas_file = gwas_files
)

# ------------------------------------------------------------
# 10. RUN DISEASE LOOP
# ------------------------------------------------------------
importance_rows <- list()
overall_rows <- list()
supp_rows <- list()
skip_rows <- list()
diag_rows <- list()

diseases_common <- intersect(unique(latent_all$Disease), gwas_tbl$Disease)

for (d in diseases_common) {
  message("Processing disease: ", d)

  lat_d <- latent_all %>% filter(Disease == d)

  gwas_file_d <- gwas_tbl %>%
    filter(Disease == d) %>%
    slice(1) %>%
    pull(gwas_file)

  if (length(gwas_file_d) == 0 || is.na(gwas_file_d)) {
    skip_rows[[d]] <- tibble(Disease = d, reason = "Missing GWAS file")
    next
  }

  bim_lookup_d <- load_bim_lookup_for_disease(bim_root, d)

  gwas_d <- tryCatch(
    {
      read_gwas_robust(gwas_file_d) %>%
        resolve_gwas_snp_to_rsid(bim_lookup_d) %>%
        filter(!is.na(P), !is.na(rsid)) %>%
        arrange(P) %>%
        distinct(rsid, .keep_all = TRUE)
    },
    error = function(e) {
      warning("Skipping disease ", d, " because GWAS parsing failed: ", e$message)
      return(NULL)
    }
  )

  if (is.null(gwas_d) || nrow(gwas_d) < 1) {
    skip_rows[[d]] <- tibble(Disease = d, reason = "No usable GWAS SNPs after BIM mapping")
    next
  }

  gwas_ranked_snps <- gwas_d %>% pull(rsid) %>% unique()

  lv_levels <- lat_d %>%
    distinct(LV, LV_num) %>%
    arrange(LV_num) %>%
    pull(LV)

  disease_selected_union <- character(0)
  disease_any <- FALSE

  for (lv in lv_levels) {
    lv_candidates <- lat_d %>%
      filter(LV == lv) %>%
      count(SNP, sort = TRUE, name = "freq") %>%
      arrange(desc(freq), SNP) %>%
      pull(SNP) %>%
      unique()

    if (length(lv_candidates) < 1) next

    path_df <- greedy_select_lv(
      candidate_snps = lv_candidates,
      gwas_ranked_snps = gwas_ranked_snps,
      disease = d,
      lv = lv,
      cs2g_tbl = cs2g,
      disgenet_tbl = disgenet_sets,
      drug_tbl = drug_sets,
      max_k = MAX_K_PER_LV
    )

    if (is.null(path_df) || nrow(path_df) == 0) next

    disease_any <- TRUE

    imp_df <- path_df %>%
      arrange(step) %>%
      mutate(
        prev_objective = lag(objective, default = 0),
        empirical_importance = objective - prev_objective
      ) %>%
      select(
        Disease, LV, step, selected_SNP, empirical_importance,
        objective, Delta_DisGeNET, Delta_Drug,
        LV_DisGeNET, GWAS_DisGeNET, LV_Drug, GWAS_Drug
      )

    importance_rows[[paste(d, lv, sep = "_")]] <- imp_df

    best_dis <- path_df %>%
      arrange(desc(Delta_DisGeNET), desc(Delta_Drug), step) %>%
      slice(1)

    supp_rows[[paste(d, lv, sep = "_")]] <- tibble(
      Disease = d,
      LV = lv,
      max_delta_DisGeNET = best_dis$Delta_DisGeNET,
      best_k_DisGeNET = best_dis$k,
      max_delta_Drug = best_dis$Delta_Drug,
      best_k_Drug = best_dis$k
    )

    lv_selected_snps <- imp_df %>%
      arrange(step) %>%
      pull(selected_SNP) %>%
      unique()

    disease_selected_union <- union(disease_selected_union, lv_selected_snps)
  }

  if (!disease_any) {
    skip_rows[[d]] <- tibble(Disease = d, reason = "No LV produced a valid greedy path")
    next
  }

  K_union <- length(disease_selected_union)

  if (K_union < MIN_FINAL_UNION_SIZE) {
    skip_rows[[d]] <- tibble(Disease = d, reason = "Final representation union too small")
    next
  }

  gwas_top_snps <- gwas_ranked_snps[seq_len(min(K_union, length(gwas_ranked_snps)))]
  K_final <- length(gwas_top_snps)

  if (K_final < 1) {
    skip_rows[[d]] <- tibble(Disease = d, reason = "Matched GWAS budget is zero")
    next
  }

  repr_final_snps <- disease_selected_union[seq_len(min(K_final, length(disease_selected_union)))]
  K_final <- length(repr_final_snps)

  repr_genes <- cs2g %>%
    filter(SNP %in% repr_final_snps) %>%
    pull(GENE) %>%
    unique()

  gwas_genes <- cs2g %>%
    filter(SNP %in% gwas_top_snps[seq_len(K_final)]) %>%
    pull(GENE) %>%
    unique()

  repr_dis_count  <- score_gene_set(repr_genes, disgenet_sets, d)
  gwas_dis_count  <- score_gene_set(gwas_genes, disgenet_sets, d)
  repr_drug_count <- score_gene_set(repr_genes, drug_sets, d)
  gwas_drug_count <- score_gene_set(gwas_genes, drug_sets, d)

  overall_rows[[d]] <- tibble(
    Disease = d,
    K_representation_union = length(disease_selected_union),
    K_final = K_final,
    Representation_DisGeNET = repr_dis_count,
    GWAS_DisGeNET = gwas_dis_count,
    Delta_DisGeNET = repr_dis_count - gwas_dis_count,
    Representation_Drug = repr_drug_count,
    GWAS_Drug = gwas_drug_count,
    Delta_Drug = repr_drug_count - gwas_drug_count,
    Representation_mapped_genes = length(repr_genes),
    GWAS_mapped_genes = length(gwas_genes)
  )

  diag_rows[[d]] <- tibble(
    Disease = d,
    n_lvs = n_distinct(lat_d$LV),
    max_k_per_lv = MAX_K_PER_LV,
    representation_union_size = length(disease_selected_union),
    gwas_mappable_snp_count = length(gwas_ranked_snps),
    final_matched_budget = K_final
  )
}

importance_df <- bind_rows(importance_rows)
overall_df    <- bind_rows(overall_rows)
supp_df       <- bind_rows(supp_rows)
skip_df       <- bind_rows(skip_rows)
diag_df       <- bind_rows(diag_rows)

if (nrow(overall_df) == 0) {
  stop("No overall dynamic comparisons could be completed.")
}

# ------------------------------------------------------------
# 11. ORDERING
# ------------------------------------------------------------
disease_order <- overall_df %>%
  arrange(Delta_DisGeNET) %>%
  pull(Disease)

lv_order <- supp_df %>%
  distinct(LV) %>%
  mutate(LV_num = as.numeric(gsub("LV", "", LV))) %>%
  arrange(LV_num) %>%
  pull(LV)

overall_df <- overall_df %>%
  mutate(Disease = factor(Disease, levels = disease_order))

supp_df <- supp_df %>%
  mutate(
    Disease = factor(Disease, levels = disease_order),
    LV = factor(LV, levels = lv_order)
  )

summary_supp <- supp_df %>%
  group_by(Disease) %>%
  summarise(
    prop_better_dis = mean(max_delta_DisGeNET > 0, na.rm = TRUE),
    prop_better_drug = mean(max_delta_Drug > 0, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(Disease = factor(Disease, levels = disease_order))

abs_long <- overall_df %>%
  select(
    Disease,
    Representation_DisGeNET,
    GWAS_DisGeNET,
    Representation_Drug,
    GWAS_Drug
  ) %>%
  pivot_longer(
    cols = -Disease,
    names_to = "Measure",
    values_to = "Count"
  ) %>%
  mutate(
    Category = case_when(
      grepl("DisGeNET", Measure) ~ "DisGeNET",
      TRUE ~ drug_label
    ),
    Method = case_when(
      grepl("^Representation_", Measure) ~ "Representation",
      TRUE ~ "GWAS"
    ),
    Category = factor(Category, levels = c("DisGeNET", drug_label)),
    Method = factor(Method, levels = c("GWAS", "Representation"))
  )

# ------------------------------------------------------------
# 12. THEME
# ------------------------------------------------------------
theme_journal <- function() {
  theme_minimal(base_size = 13, base_family = "sans") +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_line(colour = "grey88", linewidth = 0.4),
      axis.line = element_line(colour = "black", linewidth = 0.45),
      axis.text = element_text(colour = "black", size = 10.5),
      axis.title = element_text(colour = "black", size = 12, face = "bold"),
      plot.title = element_text(size = 14.5, face = "bold", hjust = 0, margin = margin(b = 4)),
      plot.subtitle = element_text(size = 10, colour = "grey30", hjust = 0, lineheight = 1.05, margin = margin(b = 8)),
      plot.caption = element_text(size = 9.5, colour = "grey30", hjust = 0),
      legend.title = element_text(size = 10.5, face = "bold"),
      legend.text = element_text(size = 10),
      plot.margin = margin(12, 10, 10, 10)
    )
}

# ------------------------------------------------------------
# 13. MAIN FIGURE
# ------------------------------------------------------------
max_abs_dis <- max(abs(overall_df$Delta_DisGeNET), na.rm = TRUE)
if (!is.finite(max_abs_dis) || max_abs_dis == 0) max_abs_dis <- 1

p_main_a <- overall_df %>%
  ggplot(aes(x = Delta_DisGeNET, y = Disease, fill = Delta_DisGeNET)) +
  geom_col(width = 0.72) +
  geom_text(
    aes(label = Delta_DisGeNET),
    hjust = ifelse(overall_df$Delta_DisGeNET >= 0, -0.12, 1.12),
    size = 3.2
  ) +
  scale_fill_gradient2(
    low = "#3B4CC0",
    mid = "white",
    high = "#B40426",
    midpoint = 0,
    limits = c(-max_abs_dis, max_abs_dis),
    name = "Δ DisGeNET genes\n(Representation - GWAS)"
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.08, 0.12))) +
  labs(
    title = "a  Overall gain over GWAS in disease-relevant gene recovery",
    subtitle = "Representation is built dynamically by selecting SNPs that most improve matched-budget recovery over GWAS.",
    x = "Δ DisGeNET genes",
    y = "Disease"
  ) +
  theme_journal()

max_abs_drug <- max(abs(overall_df$Delta_Drug), na.rm = TRUE)
if (!is.finite(max_abs_drug) || max_abs_drug == 0) max_abs_drug <- 1

p_main_b <- overall_df %>%
  ggplot(aes(x = Delta_Drug, y = Disease, fill = Delta_Drug)) +
  geom_col(width = 0.72) +
  geom_text(
    aes(label = Delta_Drug),
    hjust = ifelse(overall_df$Delta_Drug >= 0, -0.12, 1.12),
    size = 3.2
  ) +
  scale_fill_gradient2(
    low = "#3B4CC0",
    mid = "white",
    high = "#B40426",
    midpoint = 0,
    limits = c(-max_abs_drug, max_abs_drug),
    name = paste0("Δ ", drug_label, " genes\n(Representation - GWAS)")
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.08, 0.12))) +
  labs(
    title = glue("b  Overall gain over GWAS in {drug_label} recovery"),
    subtitle = "Same matched-SNP comparison, evaluated against target-relevant genes.",
    x = glue("Δ {drug_label} genes"),
    y = NULL
  ) +
  theme_journal() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

p_main_c <- abs_long %>%
  filter(Category == "DisGeNET") %>%
  ggplot(aes(x = Count, y = Disease, fill = Method)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  scale_fill_manual(values = c("GWAS" = "#7A7A7A", "Representation" = "#54A24B")) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(
    title = "c  Absolute disease-relevant gene recovery",
    subtitle = "Counts of DisGeNET genes recovered by GWAS and by the dynamically selected representation.",
    x = "Recovered DisGeNET genes",
    y = "Disease"
  ) +
  theme_journal()

p_main_d <- abs_long %>%
  filter(Category == drug_label) %>%
  ggplot(aes(x = Count, y = Disease, fill = Method)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  scale_fill_manual(values = c("GWAS" = "#7A7A7A", "Representation" = "#54A24B")) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(
    title = glue("d  Absolute {drug_label} gene recovery"),
    subtitle = glue("Counts of {drug_label} genes recovered by GWAS and by the dynamically selected representation."),
    x = glue("Recovered {drug_label} genes"),
    y = NULL
  ) +
  theme_journal() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

main_plot <- (p_main_a | p_main_b) / (p_main_c | p_main_d) +
  plot_layout(guides = "collect", widths = c(1, 1), heights = c(1.05, 1)) &
  theme(
    legend.position = "right",
    plot.background = element_rect(fill = "white", colour = NA)
  )

# ------------------------------------------------------------
# 14. SUPPLEMENTARY FIGURE
# ------------------------------------------------------------
max_abs_supp_dis <- max(abs(supp_df$max_delta_DisGeNET), na.rm = TRUE)
if (!is.finite(max_abs_supp_dis) || max_abs_supp_dis == 0) max_abs_supp_dis <- 1

p_supp_a <- ggplot(supp_df, aes(x = LV, y = Disease, fill = max_delta_DisGeNET)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = max_delta_DisGeNET), size = 2.9, fontface = "bold") +
  scale_fill_gradient2(
    low = "#3B4CC0",
    mid = "white",
    high = "#B40426",
    midpoint = 0,
    limits = c(-max_abs_supp_dis, max_abs_supp_dis),
    name = "Max Δ DisGeNET"
  ) +
  labs(
    title = "a  Best per-LV gain over GWAS in disease-relevant recovery",
    subtitle = "Each cell shows the strongest gain achieved by an LV during dynamic SNP selection.",
    x = "Latent variable",
    y = "Disease"
  ) +
  theme_journal()

p_supp_b <- ggplot(supp_df, aes(x = LV, y = Disease, fill = best_k_DisGeNET)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = best_k_DisGeNET), size = 2.9, fontface = "bold") +
  scale_fill_viridis_c(option = "C", name = expression(k^"*")) +
  labs(
    title = "b  Optimal selected SNP budget per LV",
    subtitle = "The selected budget is the point where each LV most outperforms GWAS in DisGeNET recovery.",
    x = "Latent variable",
    y = NULL
  ) +
  theme_journal() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

max_abs_supp_drug <- max(abs(supp_df$max_delta_Drug), na.rm = TRUE)
if (!is.finite(max_abs_supp_drug) || max_abs_supp_drug == 0) max_abs_supp_drug <- 1

p_supp_c <- ggplot(supp_df, aes(x = LV, y = Disease, fill = max_delta_Drug)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = max_delta_Drug), size = 2.9, fontface = "bold") +
  scale_fill_gradient2(
    low = "#3B4CC0",
    mid = "white",
    high = "#B40426",
    midpoint = 0,
    limits = c(-max_abs_supp_drug, max_abs_supp_drug),
    name = paste0("Max Δ ", drug_label)
  ) +
  labs(
    title = glue("c  Best per-LV gain over GWAS in {drug_label} recovery"),
    subtitle = "Each cell shows the strongest gain achieved by an LV during dynamic SNP selection.",
    x = "Latent variable",
    y = "Disease"
  ) +
  theme_journal()

plot_df_d <- summary_supp %>%
  select(Disease, prop_better_dis, prop_better_drug) %>%
  pivot_longer(
    cols = c(prop_better_dis, prop_better_drug),
    names_to = "Metric",
    values_to = "Fraction"
  ) %>%
  mutate(
    Metric = factor(
      Metric,
      levels = c("prop_better_dis", "prop_better_drug"),
      labels = c("DisGeNET", drug_label)
    )
  )

p_supp_d <- ggplot(plot_df_d, aes(x = Fraction, y = Disease, fill = Metric)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  scale_fill_manual(values = c("DisGeNET" = "#666666", !!drug_label := "#54A24B")) +
  scale_x_continuous(
    limits = c(0, 1.05),
    breaks = seq(0, 1, 0.25),
    labels = percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "d  Fraction of latent variables that ever outperform GWAS",
    subtitle = "A disease contributes positively when an LV achieves better matched-budget recovery than GWAS.",
    x = "Fraction of LVs beating GWAS",
    y = NULL
  ) +
  theme_journal() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

supp_plot <- (p_supp_a | p_supp_b) / (p_supp_c | p_supp_d) +
  plot_layout(guides = "collect", widths = c(1, 1), heights = c(1, 1)) &
  theme(
    legend.position = "right",
    plot.background = element_rect(fill = "white", colour = NA)
  )

# ------------------------------------------------------------
# 15. SAVE
# ------------------------------------------------------------
pdf_device_to_use <- if (capabilities("cairo")) cairo_pdf else "pdf"

ggsave(
  filename = paste0(out_prefix_main, ".pdf"),
  plot = main_plot,
  width = 16,
  height = 11,
  units = "in",
  dpi = 600,
  device = pdf_device_to_use,
  bg = "white"
)

ggsave(
  filename = paste0(out_prefix_main, ".png"),
  plot = main_plot,
  width = 16,
  height = 11,
  units = "in",
  dpi = 600,
  bg = "white"
)

ggsave(
  filename = paste0(out_prefix_main, ".tiff"),
  plot = main_plot,
  width = 16,
  height = 11,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)

ggsave(
  filename = paste0(out_prefix_supp, ".pdf"),
  plot = supp_plot,
  width = 16,
  height = 11,
  units = "in",
  dpi = 600,
  device = pdf_device_to_use,
  bg = "white"
)

ggsave(
  filename = paste0(out_prefix_supp, ".png"),
  plot = supp_plot,
  width = 16,
  height = 11,
  units = "in",
  dpi = 600,
  bg = "white"
)

ggsave(
  filename = paste0(out_prefix_supp, ".tiff"),
  plot = supp_plot,
  width = 16,
  height = 11,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)

# ------------------------------------------------------------
# 16. EXPORT TABLES
# ------------------------------------------------------------
write_csv(importance_df, file.path(out_dir, "dynamic_empirical_snp_importance.csv"))
write_csv(overall_df,    file.path(out_dir, "dynamic_overall_representation_vs_gwas.csv"))
write_csv(supp_df,       file.path(out_dir, "dynamic_perLV_vs_gwas.csv"))
write_csv(diag_df,       file.path(out_dir, "dynamic_selection_diagnostics.csv"))

if (nrow(skip_df) > 0) {
  write_csv(skip_df, file.path(out_dir, "dynamic_selection_skipped_diseases.csv"))
}

message("Done.")
