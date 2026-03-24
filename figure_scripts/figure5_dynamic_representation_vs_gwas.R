#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
  library(patchwork)
  library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg_value <- function(flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default)
  if (idx == length(args)) return(default)
  args[idx + 1]
}

latent_root   <- get_arg_value("--latent_root", "examples/manuscript_summary_tables/latent")
gwas_root     <- get_arg_value("--gwas_root", "examples/manuscript_summary_tables/gwas")
bim_root      <- get_arg_value("--bim_root", "examples/manuscript_summary_tables/bim")
cs2g_file     <- get_arg_value("--cs2g_file", "examples/manuscript_summary_tables/combined_cS2G.tsv")
disgenet_file <- get_arg_value("--disgenet_file", "examples/manuscript_summary_tables/consolidated.tsv")
drug_file     <- get_arg_value("--drug_target_file", "examples/manuscript_summary_tables/drug_target_gene_sets.tsv")
out_dir       <- get_arg_value("--out_dir", "Figure5_dynamic_representation_vs_gwas")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

extract_disease_from_latent <- function(path) {
  nm <- basename(path)
  sub("^latent_gene_table\\.ALL_SAMPLES\\.long_([A-Za-z0-9]+)_LD.*\\.csv$", "\\1", nm)
}

extract_disease_from_gwas <- function(path) {
  nm <- basename(path)
  sub("^([A-Za-z0-9]+)_gwas\\.(assoc|qassoc)$", "\\1", nm)
}

score_gene_set <- function(genes, ref_tbl, disease) {
  ref <- ref_tbl %>% filter(Disease == disease) %>% pull(GENE) %>% unique()
  sum(unique(genes) %in% ref)
}

read_gwas <- function(path) {
  df <- read.table(path, header = TRUE, stringsAsFactors = FALSE, sep = "", check.names = FALSE)
  if (!"SNP" %in% names(df) || !"P" %in% names(df)) return(tibble())
  as_tibble(df) %>% mutate(SNP = as.character(SNP), P = as.numeric(P)) %>% filter(!is.na(P)) %>% arrange(P)
}

if (!file.exists(cs2g_file) || !file.exists(disgenet_file)) {
  stop("Required cs2g/disgenet files not found.")
}

cs2g <- read_tsv(cs2g_file, show_col_types = FALSE) %>% transmute(SNP = as.character(SNP), GENE = toupper(as.character(GENE))) %>% distinct()
disgenet <- read_tsv(disgenet_file, show_col_types = FALSE) %>% transmute(Disease = as.character(doid_name), GENE = toupper(as.character(geneSymbol))) %>% distinct()

if (file.exists(drug_file)) {
  drug_sets <- read_tsv(drug_file, show_col_types = FALSE) %>% transmute(Disease = as.character(Disease), GENE = toupper(as.character(GENE))) %>% distinct()
} else {
  drug_sets <- tibble(Disease = character(), GENE = character())
}

latent_files <- list.files(latent_root, pattern = "^latent_gene_table\\.ALL_SAMPLES\\.long_.*\\.csv$", full.names = TRUE, recursive = TRUE)
gwas_files <- list.files(gwas_root, pattern = "(_gwas\\.(assoc|qassoc))$", full.names = TRUE)

if (length(latent_files) == 0 || length(gwas_files) == 0) {
  stop("No latent or GWAS files found.")
}

latent_all <- purrr::map_dfr(latent_files, function(fp) {
  disease <- extract_disease_from_latent(fp)
  read_csv(fp, show_col_types = FALSE) %>% mutate(Disease = disease)
})

gwas_tbl <- tibble(Disease = vapply(gwas_files, extract_disease_from_gwas, character(1)), gwas_file = gwas_files)

rows <- list()
for (d in intersect(unique(latent_all$Disease), gwas_tbl$Disease)) {
  lat_d <- latent_all %>% filter(Disease == d)
  gwas_file_d <- gwas_tbl %>% filter(Disease == d) %>% slice(1) %>% pull(gwas_file)
  gwas_d <- read_gwas(gwas_file_d)
  if (nrow(gwas_d) == 0) next

  repr_snps <- lat_d %>% pull(SNP_ID) %>% as.character() %>% unique()
  K <- min(length(repr_snps), nrow(gwas_d))
  if (K < 1) next
  repr_snps <- repr_snps[seq_len(K)]
  gwas_snps <- gwas_d$SNP[seq_len(K)]

  repr_genes <- cs2g %>% filter(SNP %in% repr_snps) %>% pull(GENE) %>% unique()
  gwas_genes <- cs2g %>% filter(SNP %in% gwas_snps) %>% pull(GENE) %>% unique()

  rows[[d]] <- tibble(
    Disease = d,
    K_final = K,
    Representation_DisGeNET = score_gene_set(repr_genes, disgenet, d),
    GWAS_DisGeNET = score_gene_set(gwas_genes, disgenet, d),
    Representation_Drug = score_gene_set(repr_genes, drug_sets, d),
    GWAS_Drug = score_gene_set(gwas_genes, drug_sets, d)
  ) %>% mutate(
    Delta_DisGeNET = Representation_DisGeNET - GWAS_DisGeNET,
    Delta_Drug = Representation_Drug - GWAS_Drug
  )
}

overall_df <- bind_rows(rows)
if (nrow(overall_df) == 0) stop("No dynamic comparisons could be computed.")
write_csv(overall_df, file.path(out_dir, "dynamic_overall_representation_vs_gwas.csv"))

disease_order <- overall_df %>% arrange(Delta_DisGeNET) %>% pull(Disease)
overall_df <- overall_df %>% mutate(Disease = factor(Disease, levels = disease_order))

abs_long <- overall_df %>%
  select(Disease, Representation_DisGeNET, GWAS_DisGeNET, Representation_Drug, GWAS_Drug) %>%
  pivot_longer(cols = -Disease, names_to = "Measure", values_to = "Count") %>%
  mutate(
    Category = ifelse(grepl("DisGeNET", Measure), "DisGeNET", "Drug target"),
    Method = ifelse(grepl("^Representation", Measure), "Representation", "GWAS")
  )

theme_journal <- function() {
  theme_minimal(base_size = 13, base_family = "sans") +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_line(colour = "grey88", linewidth = 0.4),
      axis.line = element_line(colour = "black", linewidth = 0.45),
      axis.text = element_text(colour = "black", size = 10.5),
      axis.title = element_text(colour = "black", size = 12, face = "bold"),
      plot.title = element_text(size = 14.5, face = "bold", hjust = 0),
      legend.title = element_blank(),
      legend.text = element_text(size = 10)
    )
}

max_abs_dis <- max(abs(overall_df$Delta_DisGeNET), na.rm = TRUE); if (!is.finite(max_abs_dis) || max_abs_dis == 0) max_abs_dis <- 1
max_abs_drug <- max(abs(overall_df$Delta_Drug), na.rm = TRUE); if (!is.finite(max_abs_drug) || max_abs_drug == 0) max_abs_drug <- 1

p1 <- ggplot(overall_df, aes(x = Delta_DisGeNET, y = Disease, fill = Delta_DisGeNET)) +
  geom_col(width = 0.72) +
  geom_text(aes(label = Delta_DisGeNET), hjust = ifelse(overall_df$Delta_DisGeNET >= 0, -0.12, 1.12), size = 3.2) +
  scale_fill_gradient2(low = "#3B4CC0", mid = "white", high = "#B40426", midpoint = 0, limits = c(-max_abs_dis, max_abs_dis)) +
  labs(title = "a  Overall gain over GWAS in disease-relevant gene recovery", x = "Δ DisGeNET genes", y = "Disease") + theme_journal()

p2 <- ggplot(overall_df, aes(x = Delta_Drug, y = Disease, fill = Delta_Drug)) +
  geom_col(width = 0.72) +
  geom_text(aes(label = Delta_Drug), hjust = ifelse(overall_df$Delta_Drug >= 0, -0.12, 1.12), size = 3.2) +
  scale_fill_gradient2(low = "#3B4CC0", mid = "white", high = "#B40426", midpoint = 0, limits = c(-max_abs_drug, max_abs_drug)) +
  labs(title = "b  Overall gain over GWAS in target recovery", x = "Δ target genes", y = NULL) + theme_journal() + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

p3 <- abs_long %>% filter(Category == "DisGeNET") %>% ggplot(aes(x = Count, y = Disease, fill = Method)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  scale_fill_manual(values = c("GWAS" = "#7A7A7A", "Representation" = "#54A24B")) +
  labs(title = "c  Absolute disease-relevant gene recovery", x = "Recovered DisGeNET genes", y = "Disease") + theme_journal()

p4 <- abs_long %>% filter(Category == "Drug target") %>% ggplot(aes(x = Count, y = Disease, fill = Method)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  scale_fill_manual(values = c("GWAS" = "#7A7A7A", "Representation" = "#54A24B")) +
  labs(title = "d  Absolute target gene recovery", x = "Recovered target genes", y = NULL) + theme_journal() + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

main_plot <- (p1 | p2) / (p3 | p4) + plot_layout(guides = "collect") & theme(legend.position = "right")

ggsave(file.path(out_dir, "Figure_DynamicRepresentation_vs_GWAS.pdf"), main_plot, width = 16, height = 11, units = "in", dpi = 600)
ggsave(file.path(out_dir, "Figure_DynamicRepresentation_vs_GWAS.png"), main_plot, width = 16, height = 11, units = "in", dpi = 600, bg = "white")

cat("Done.\n")
