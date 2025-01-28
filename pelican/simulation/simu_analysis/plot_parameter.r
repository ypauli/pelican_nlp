######################################
## 0) Specify your parameter of interest and file path
######################################
paramName <- "target_length"
paramCol <- "varied_param_value"
filePath <- "/home/ubuntu/emilia/PELICAN/pelican/simulation/simu_analysis/data/"

# Define file names
fileA <- paste0(filePath, paramName, "-A.csv")
fileB <- paste0(filePath, paramName, "-B.csv")

# Define bin cut points for your parameter
# Example for temperature in [0..10]:
# paramBreaks <- c(0, 2, 4, 6, 8, 10) # Example for "temperature"
# paramBreaks <- c(0.2, 0.4, 0.6, 0.8, 1)  # Example for "sampling"
paramBreaks <- c(0, 40, 80, 120, 160, 200)  # Example for "context_span"
# paramBreaks <- c(0, 40, 80, 120, 160, 200)  # Example for "target_length"

# Load libraries
library(ggplot2)
library(dplyr)
library(reshape2)
library(cowplot)

######################################
## 1) Read data 
######################################
dfA <- read.csv(fileA)
dfB <- read.csv(fileB)

######################################
## 2) Helper function for group scaling
##    Avoid scale() if group has < 2 rows
######################################
safe_scale <- function(x) {
  if (length(unique(x)) < 2) {
    # If there's only one unique value, return the same numeric but as NA or 0
    # In this example, let's return 0 so that single-row groups remain visible
    return(rep(0, length(x)))
  } else {
    return(as.numeric(scale(x)))
  }
}

######################################
## 3) Figure 4A Data: dfA_melt
##    (All word-pairs & consecutive words)
######################################
dfA_melt <- dfA %>%
  mutate(paramValue = .data[[paramCol]]) %>%
  # Keep only columns needed
  select(paramValue, prompt_number, avg_consec, avg_all_pairs) %>%
  # Reshape
  melt(
    id.vars       = c("paramValue", "prompt_number"),
    measure.vars  = c("avg_consec", "avg_all_pairs"),
    variable.name = "variable", 
    value.name    = "value"
  ) %>%
  # If you truly want to remove NA but keep 0, do filter(!is.na(value)).
  # If zero is invalid, do filter(!is.na(value) & value != 0).
  filter(!is.na(value) & value != 0) %>%
  mutate(
    variable = ifelse(variable == "avg_consec", 
                      "Consecutive words", 
                      "All word-pairs")
  ) %>%
  group_by(prompt_number, variable) %>%
  mutate(value2 = safe_scale(value)) %>%
  ungroup() %>%
  # Bin paramValue directly:
  mutate(
    gParam = cut(paramValue, breaks = paramBreaks, include.lowest=TRUE)
  )

# Convert gParam to a factor with desired labels (optional):
dfA_melt$gParam <- factor(dfA_melt$gParam)

# Correlation with paramValue
dfA_cor <- dfA_melt %>%
  group_by(variable) %>%
  summarize(
    cor   = cor(paramValue, value2, method="spearman", use="complete.obs"),
    label = paste0("R = ", round(cor, 2))
  )

######################################
## 4) Figure 4A
######################################
fig4A <- ggplot(dfA_melt, aes(x = gParam, y = value2)) +
  # Mean across *all prompts* in black
  stat_summary(fun = "mean", color = "black", size = 1, aes(group = 1)) +
  stat_summary(fun = "mean", color = "black", size = 2, geom = "line", linetype = "solid", aes(group = 1)) +
  # Facet by variable
  facet_grid(~variable, scales = "free") +
  # Per-prompt lines
  stat_summary(aes(color = factor(prompt_number), group = factor(prompt_number)),
               fun = "mean", geom = "line", alpha = 0.5) +
  theme_bw() +
  # Remove ylim(-1,1) to see if data is outside that range
  xlab(paramName) +
  ylab("Normalized semantic distance") +
  ggtitle("A) Average semantic distances between words") +
  theme(
    axis.title    = element_text(size=14),
    axis.text     = element_text(size=12),
    strip.text    = element_text(size=12),
    plot.title    = element_text(size=16),
    legend.text   = element_text(size=11),
    legend.title  = element_text(size=14)
  ) +
  geom_text(
    data     = dfA_cor,
    aes(x    = 1.5, y = Inf, vjust = 1.5, label = label), 
    color    = "blue",
    size     = 5,
    fontface = "italic"
  )

######################################
## 5) Figure 4B Data: dfB_melt
##    (Distances from prompt)
######################################
dfB_melt <- dfB %>%
  mutate(paramValue = .data[[paramCol]]) %>%
  select(paramValue, prompt_number, avg_prompt_cosine, avg_prompt_wmd) %>%
  melt(
    id.vars       = c("paramValue", "prompt_number"),
    measure.vars  = c("avg_prompt_cosine", "avg_prompt_wmd"),
    variable.name = "variable", 
    value.name    = "value"
  ) %>%
  filter(!is.na(value)) %>%
  mutate(
    variable = case_when(
      variable == "avg_prompt_cosine" ~ "Average of words",
      variable == "avg_prompt_wmd"    ~ "Word mover's distance"
    )
  ) %>%
  group_by(prompt_number, variable) %>%
  mutate(value2 = safe_scale(value)) %>%
  ungroup() %>%
  mutate(
    gParam = cut(paramValue, breaks = paramBreaks, include.lowest=TRUE)
  )

dfB_melt$gParam <- factor(dfB_melt$gParam)

dfB_cor <- dfB_melt %>%
  group_by(variable) %>%
  summarize(
    cor   = cor(paramValue, value2, method="spearman", use="complete.obs"),
    label = paste0("R = ", round(cor, 2))
  )

######################################
## 6) Figure 4B
######################################
fig4B <- ggplot(dfB_melt, aes(x = gParam, y = value2)) +
  stat_summary(fun = "mean", color = "black", size = 1, aes(group=1)) +
  stat_summary(fun = "mean", color = "black", size = 2, geom = "line", linetype = "solid", aes(group=1)) +
  facet_grid(~variable, scales = "free") +
  stat_summary(aes(color = factor(prompt_number), group = factor(prompt_number)),
               fun = "mean", geom = "line", alpha=0.5) +
  theme_bw() +
  xlab(paramName) +
  ylab("Normalized semantic distance") +
  ggtitle("B) Average semantic distances from the prompt") +
  theme(
    axis.title       = element_text(size=14),
    axis.text        = element_text(size=12),
    strip.text       = element_text(size=12),
    legend.position  = "right",
    plot.title       = element_text(size=16)
  ) +
  geom_text(
    data     = dfB_cor,
    aes(x    = 1.5, y = Inf, vjust=1.5, label=label),
    color    = "blue",
    size     = 5,
    fontface = "italic"
  )

######################################
## 7) Figure 4C Data: dfB_tangential
##    (Tangentiality across sentences)
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
  filter(!is.na(value)) %>%
  mutate(
    variable = case_when(
      variable == "avg_prompt_cosine" ~ "Average of words",
      variable == "avg_prompt_wmd"    ~ "Word mover's distance"
    )
  ) %>%
  group_by(prompt_number, variable) %>%
  mutate(value2 = safe_scale(value)) %>%
  ungroup() %>%
  mutate(
    gParam = cut(paramValue, breaks = paramBreaks, include.lowest=TRUE),
    sent   = sentence_number
  )

dfB_tangential$gParam <- factor(dfB_tangential$gParam)

######################################
## 8) Figure 4C
######################################
fig4C <- ggplot(dfB_tangential, aes(x = sent, y = value2, 
                                    color = gParam, group = gParam)) +
  stat_summary(fun = "mean", geom = "point", na.rm=TRUE) +
  stat_summary(fun = "mean", geom = "line",  na.rm=TRUE) +
  facet_grid(~variable) +
  theme_bw() +
  xlab("Sentence #") +
  ylab("Normalized semantic distance") +
  # No forced y-limits; see if data appear
  scale_color_viridis_d(option="C", end=0.8) +
  labs(color="Parameter bin") +
  ggtitle("C) Semantic distance from prompt over sentences") +
  theme(
    axis.title      = element_text(size=14),
    axis.text       = element_text(size=12),
    strip.text      = element_text(size=12),
    legend.position = "top",
    legend.text     = element_text(size=12),
    legend.title    = element_text(size=12),
    plot.title      = element_text(size=16)
  )

######################################
## 9) Combine all
######################################
final_plot <- plot_grid(
  fig4A,
  fig4B,
  fig4C,
  nrow         = 3,
  align        = "v",
  axis         = "lr",
  rel_heights  = c(0.32, 0.32, 0.36)
)

print(final_plot)