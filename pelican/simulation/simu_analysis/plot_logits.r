######################################
## 0) Specify your parameters and file path
######################################
paramNames <- c("temperature", "sampling", "context_span", "target_length")
paramCol <- "varied_param_value"
filePath <- "/home/ubuntu/emilia/data_unif_full-range/"

# Define file name
fileA <- paste0(filePath, "target_length-A.csv")

# Define bin cut points for parameters
paramBreaks <- list(
  temperature = c(0, 2, 4, 6, 8, 10),
  sampling = c(0.2, 0.4, 0.6, 0.8, 1),
  context_span = c(0, 40, 80, 120, 160, 200),
  target_length = c(0, 40, 80, 120, 160, 200)
)

# Load libraries
library(ggplot2)
library(dplyr)
library(reshape2)
library(cowplot)
library(tidyr)
library(patchwork)



######################################
## 1) Read data 
######################################
# Load data
# data <- read.csv(fileA)

# Check for missing values
# data <- data %>% drop_na()

######################################
## 2) Helper function for group scaling
######################################
safe_scale <- function(x) {
  if (length(unique(na.omit(x))) < 2) {
    return(rep(0, length(x)))
  } else {
    return(as.numeric(scale(x)))
  }
}

######################################
## 3) Process data and generate plots
######################################
metrics <- c("avg_entropy_per_section", "avg_perplexity_per_section")
plot_list <- list()

for (param in paramNames) {

  fileA <- paste0(filePath, param, "-A.csv")
  data <- read.csv(fileA)
  # Check for missing values
  data <- data %>% drop_na()

  for (metric in metrics) {
    
    df_melt <- data %>%
      mutate(paramValue = .data[[paramCol]]) %>%
      select(paramValue, prompt_number, all_of(metric)) %>%
      melt(
        id.vars       = c("paramValue", "prompt_number"),
        measure.vars  = metric,
        variable.name = "variable",
        value.name    = "value"
      ) %>%
      filter(!is.na(paramValue) & !is.na(value)) %>%
      group_by(prompt_number, variable) %>%
      mutate(value2 = safe_scale(value)) %>%
      ungroup() %>%
      mutate(
        gParam = cut(paramValue, breaks = paramBreaks[[param]], include.lowest=TRUE)
      ) %>%
      filter(!is.na(gParam))
    
    df_melt$gParam <- factor(df_melt$gParam)
    
    df_cor <- df_melt %>%
      filter(!is.na(paramValue) & !is.na(value2)) %>%
      group_by(variable) %>%
      summarize(
        cor   = ifelse(n() > 1, cor(paramValue, value2, method="spearman", use="complete.obs"), NA),
        label = paste0("R = ", round(cor, 2))
      )
    
    plot <- ggplot(df_melt, aes(x = gParam, y = value2)) +
      stat_summary(fun = "mean", color = "black", size = 1, aes(group = 1)) +
      stat_summary(fun = "mean", color = "black", size = 2, geom = "line", linetype = "solid", aes(group = 1)) +
      facet_wrap(~variable, scales = "free") +
      stat_summary(aes(color = factor(prompt_number), group = factor(prompt_number)),
                   fun = "mean", geom = "line", alpha = 0.5) +
      theme_bw() +
      xlab(param) +
      ylab("Normalized Value") +
      ggtitle(paste(metric, "vs", param)) +
      theme(
        aspect.ratio = 0.8,
        axis.title    = element_text(size=14),
        axis.text     = element_text(size=12),
        strip.text    = element_text(size=12),
        plot.title    = element_text(size=16),
        legend.text   = element_text(size=11),
        legend.title  = element_text(size=14)
      ) +
      geom_text(
        data     = df_cor,
        aes(x    = 1.5, y = Inf, vjust = 1.5, label = label), 
        color    = "blue",
        size     = 5,
        fontface = "italic"
      )
    
    plot_list[[paste(metric, param, sep = "_")]] <- plot
  }
}

######################################
## 4) Combine all plots into one image
######################################

# Combine plots into two rows, where the first row contains the plots for avg_perplexity_per_section and the second row contains the plots for avg_entropy_per_section
p1 <- plot_list[["avg_perplexity_per_section_temperature"]] + plot_list[["avg_perplexity_per_section_sampling"]] + plot_list[["avg_perplexity_per_section_context_span"]] + plot_list[["avg_perplexity_per_section_target_length"]]
p2 <- plot_list[["avg_entropy_per_section_temperature"]] + plot_list[["avg_entropy_per_section_sampling"]] + plot_list[["avg_entropy_per_section_context_span"]] + plot_list[["avg_entropy_per_section_target_length"]]

# Print the combined plot
print(plot_list[["avg_perplexity_per_section_temperature"]])