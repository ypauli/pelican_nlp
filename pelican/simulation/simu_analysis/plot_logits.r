######################################
## 0) Specify your parameters and file path
######################################
paramNames <- c("temperature", "sampling", "context_span", "target_length")
paramCol <- "varied_param_value"
filePath <- "/home/ubuntu/emilia/data_unif_old/"

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

######################################
## 1) Read data 
######################################
dfA <- read.csv(fileA, stringsAsFactors = FALSE)

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

for (metric in metrics) {
  for (param in paramNames) {
    
    df_melt <- dfA %>%
      mutate(paramValue = .data[[param]]) %>%
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
      group_by(variable) %>%
      summarize(
        cor   = cor(paramValue, value2, method="spearman", use="complete.obs"),
        label = paste0("R = ", round(cor, 2))
      )
    
    plot <- ggplot(df_melt, aes(x = gParam, y = value2)) +
      stat_summary(fun = "mean", color = "black", size = 1, aes(group = 1)) +
      stat_summary(fun = "mean", color = "black", size = 2, geom = "line", linetype = "solid", aes(group = 1)) +
      facet_grid(~variable, scales = "free") +
      stat_summary(aes(color = factor(prompt_number), group = factor(prompt_number)),
                   fun = "mean", geom = "line", alpha = 0.5) +
      theme_bw() +
      xlab(param) +
      ylab("Normalized Value") +
      ggtitle(paste(metric, "vs", param)) +
      theme(
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
if (length(plot_list) > 0) {
  final_plot <- plot_grid(plotlist = plot_list, nrow = ceiling(length(plot_list)/2), ncol = 2, align = "v")
  print(final_plot)
} else {
  print("No valid parameter and metric combinations found.")
}