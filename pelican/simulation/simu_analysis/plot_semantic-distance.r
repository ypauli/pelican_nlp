######################################
## 0) Setup: Specify parameters, file paths, and load libraries
######################################
paramName <- "context_span"
paramCol <- "varied_param_value"
filePath <- "/home/ubuntu/emilia/data_unif_constrained/"

# Define file names
fileA <- paste0(filePath, paramName, "-A.csv")
fileB <- paste0(filePath, paramName, "-B.csv")

# Define bin cut points for your parameter
paramBreaks <- c(0, 40, 80, 120, 160, 200)  # Example for "context_span" and "target_length"

# Load libraries
library(ggplot2)
library(dplyr)
library(reshape2)
library(cowplot)

######################################
## 1) Helper Functions
######################################
# If paramName is "temperature", remove values larger than 3.75
temp_filter <- function(df) {
  if (paramName == "temperature") {
    df <- df %>% filter(.data[[paramCol]] <= 3.75)
  }
  return(df)
}

######################################
## Helper function for group scaling
## Avoid scale() if group has < 2 rows
######################################
safe_scale <- function(x) {
  if (length(unique(x)) < 2) {
    return(rep(0, length(x)))
  } else {
    return(as.numeric(scale(x)))
  }
}

safe_scale_minmax <- function(x) {
  if (length(unique(x)) < 2) {
    return(rep(0, length(x)))  # If only 1 unique value, return 0
  } else {
    return(2 * ((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))) - 1)
  }
}

######################################
## 2) Data Preparation
######################################
## Read data 
dfA <- read.csv(fileA)
dfB <- read.csv(fileB)

dfA <- temp_filter(dfA)
dfB <- temp_filter(dfB)

######################################
## Figure 4A Data: dfA_melt
## (All word-pairs & consecutive words)
######################################
dfA_melt <- dfA %>%
  mutate(paramValue = .data[[paramCol]]) %>%
  select(paramValue, prompt_number, avg_consec, avg_all_pairs) %>%
  melt(
    id.vars       = c("paramValue", "prompt_number"),
    measure.vars  = c("avg_consec", "avg_all_pairs"),
    variable.name = "variable",
    value.name    = "value"
  ) %>%
  # Remove rows with NA in paramValue or value and where value == 0
  filter(!is.na(paramValue) & !is.na(value) & value != 0) %>%
  mutate(
    variable = ifelse(variable == "avg_consec",
                      "Consecutive words",
                      "All word-pairs")
  ) %>%
  group_by(prompt_number, variable) %>%
  mutate(value2 = scale.default(value)) %>%
  ungroup() %>%
  mutate(
    gParam = cut(paramValue, breaks = paramBreaks, include.lowest = TRUE)
  ) %>%
  filter(!is.na(gParam))

# Convert gParam to factor
dfA_melt$gParam <- factor(dfA_melt$gParam)

# Compute correlation per variable
dfA_cor <- dfA_melt %>%
  group_by(variable) %>%
  summarize(
    cor   = cor(paramValue, value2, method = "spearman", use = "complete.obs"),
    label = paste0("R = ", round(cor, 2))
  )

######################################
## Figure 4B Data: dfA_melt_B
## (Average sentence distances from sentences using dfA)
######################################
dfA_melt_B <- dfA %>%
  mutate(paramValue = .data[[paramCol]]) %>%
  select(paramValue, prompt_number, avg_sentence_distances, wmd_sentence_distances) %>%
  melt(
    id.vars       = c("paramValue", "prompt_number"),
    measure.vars  = c("avg_sentence_distances", "wmd_sentence_distances"),
    variable.name = "variable",
    value.name    = "value"
  ) %>%
  filter(!is.na(paramValue) & !is.na(value)) %>%
  mutate(
    variable = case_when(
      variable == "avg_sentence_distances" ~ "Average sentence distance",
      variable == "wmd_sentence_distances"   ~ "Word mover's distance"
    )
  ) %>%
  group_by(prompt_number, variable) %>%
  mutate(value2 = safe_scale(value)) %>%
  ungroup() %>%
  mutate(
    gParam = cut(paramValue, breaks = paramBreaks, include.lowest = TRUE)
  ) %>%
  filter(!is.na(gParam))

# Convert gParam to factor
dfA_melt_B$gParam <- factor(dfA_melt_B$gParam)

# Compute correlation per variable for Figure 4B
dfA_B_cor <- dfA_melt_B %>%
  group_by(variable) %>%
  summarize(
    cor   = cor(paramValue, value2, method = "spearman", use = "complete.obs"),
    label = paste0("R = ", round(cor, 2))
  )

######################################
## Figure 4C Data: dfB_tangential
## (Tangentiality across sentences)
######################################
dfB_tangential <- dfB %>%
  mutate(paramValue = .data[[paramCol]]) %>%
  select(paramValue, prompt_number, sentence_number, avg_prompt_cosine, avg_prompt_wmd) %>%
  melt(
    id.vars       = c("paramValue", "prompt_number", "sentence_number"),
    measure.vars  = c("avg_prompt_cosine", "avg_prompt_wmd"),
    variable.name = "variable",
    value.name    = "value"
  ) %>%
  filter(!is.na(paramValue) & !is.na(value)) %>%
  mutate(
    variable = case_when(
      variable == "avg_prompt_cosine" ~ "Average of words",
      variable == "avg_prompt_wmd"    ~ "Word mover's distance"
    )
  ) %>%
  group_by(prompt_number, variable) %>%
  mutate(value2 = safe_scale_minmax(value)) %>%
  ungroup() %>%
  mutate(
    gParam = cut(paramValue, breaks = paramBreaks, include.lowest = TRUE),
    sent   = sentence_number
  ) %>%
  filter(!is.na(gParam))

dfB_tangential$gParam <- factor(dfB_tangential$gParam)

# Count datapoints per sentence and gParam and merge the counts back
dfB_tangential_count <- dfB_tangential %>%
  group_by(sent, gParam, variable) %>%
  summarize(n = n(), .groups = 'drop')

dfB_tangential <- dfB_tangential %>%
  left_join(dfB_tangential_count, by = c("sent", "gParam", "variable"))

######################################
## 3) Plot Definitions
######################################
## Figure 4A
fig4A <- ggplot(dfA_melt, aes(x = gParam, y = value2)) +
  stat_summary(fun = "mean", color = "black", size = 1, aes(group = 1)) +
  stat_summary(fun = "mean", color = "black", size = 2, geom = "line", linetype = "solid", aes(group = 1)) +
  facet_grid(~variable, scales = "free") +
  stat_summary(aes(color = factor(prompt_number, labels = c("In meinem letzten Traum war", 
                                                            "Von hier aus bis zum nächsten Supermarkt gelangt man", 
                                                            "Seit letzter Woche habe", 
                                                            "Ich werde so viele Tiere aufzählen wie möglich: Pelikan,")), 
                       group = factor(prompt_number)),
               fun = "mean", geom = "line", alpha = 0.5) +
  theme_bw() +
  xlab(paramName) +
  ylab("Average semantic distance") +
  ggtitle("Average semantic distances between words") +
  theme(
    aspect.ratio      = 1,
    axis.title        = element_text(size = 14),
    axis.text         = element_text(size = 12),
    strip.text        = element_text(size = 12),
    plot.title        = element_text(size = 20),
    plot.title.position = "panel",
    legend.position   = "none",
    legend.direction  = "vertical",
    legend.text       = element_text(size = 11),
    legend.title      = element_text(size = 14)
  ) +
  geom_text(
    data     = dfA_cor,
    aes(x = 0, y = -Inf, vjust = -1, hjust = -0.5, label = label),
    color    = "blue",
    size     = 8,
    fontface = "italic"
  ) + labs(color = "Prompt")

## Figure 4B (using dfA and the new sentence distance measures)
fig4B <- ggplot(dfA_melt_B, aes(x = gParam, y = value2)) +
  stat_summary(fun = "mean", color = "black", size = 1, aes(group = 1)) +
  stat_summary(fun = "mean", color = "black", size = 2, geom = "line", linetype = "solid", aes(group = 1)) +
  facet_grid(~variable, scales = "free") +
  stat_summary(aes(color = factor(prompt_number, labels = c("In meinem letzten Traum war", 
                                                            "Von hier aus bis zum nächsten Supermarkt gelangt man", 
                                                            "Seit letzter Woche habe", 
                                                            "Ich werde so viele Tiere aufzählen wie möglich: Pelikan,")), 
                       group = factor(prompt_number)),
               fun = "mean", geom = "line", alpha = 0.5) +
  theme_bw() +
  xlab(paramName) +
  ylab("Normalized semantic distance") +
  ggtitle("Average semantic distances between sentences") +
  theme(
    aspect.ratio    = 1,
    axis.title      = element_text(size = 14),
    axis.text       = element_text(size = 12),
    strip.text      = element_text(size = 12),
    plot.title      = element_text(size = 20),
    legend.position = "right",
    legend.direction= "vertical",
    legend.text     = element_text(size = 11),
    legend.title    = element_text(size = 14)
  ) +
  geom_text(
    data     = dfA_B_cor,
    aes(x = 0, y = -Inf, vjust = -1, hjust = -0.5, label = label),
    color    = "blue",
    size     = 8,
    fontface = "italic"
  ) + labs(color = "Prompt")

## Figure 4C
fig4C <- ggplot(dfB_tangential, aes(x = sent, y = value2, color = gParam, group = gParam)) +
  stat_summary(fun = "mean", aes(size = n, alpha = n), na.rm = TRUE) +
  stat_summary(fun = "mean", geom = "line", aes(alpha = n), na.rm = TRUE) +
  facet_grid(~variable) +
  theme_bw() +
  scale_color_viridis_d(option = "C", end = 0.8) +
  scale_size(range = c(0, 0.7)) +
  scale_alpha(range = c(0.2, 1)) +
  labs(color = "Parameter Values") +
  guides(alpha = "none", size = "none") +
  ggtitle("Semantic distance from prompt over sentences") +
  xlab("Sentence #") +
  ylab("Normalized semantic distance") +
  theme(
    aspect.ratio    = 1,
    axis.title      = element_text(size = 14),
    axis.text       = element_text(size = 12),
    strip.text      = element_text(size = 12),
    legend.position = "right",
    legend.direction= "vertical",
    legend.text     = element_text(size = 12),
    legend.title    = element_text(size = 12),
    plot.title      = element_text(size = 16)
  ) +
  scale_x_continuous(limits = c(1, NA))  # x-axis starts at 1

######################################
## 4) Combine all Plots
######################################
final_plot <- plot_grid(
  fig4A, fig4B, fig4C,
  nrow         = 3,
  align        = "v",
  axis         = "l",
  rel_heights  = c(5, 5, 5)
)

print(final_plot)