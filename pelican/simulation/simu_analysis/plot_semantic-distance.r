######################################
## 0) Specify your parameter of interest and file path
######################################
paramName <- "temperature"
paramCol <- "varied_param_value"
filePath <- "/home/ubuntu/emilia/data_unif/"

# Define file names
fileA <- paste0(filePath, paramName, "-A.csv")
fileB <- paste0(filePath, paramName, "-B.csv")

# Define bin cut points for your parameter
# Example for temperature in [0..10]:
paramBreaks <- c(0, 1, 2, 3) # Example for "temperature"
# paramBreaks <- c(0.2, 0.4, 0.6, 0.8, 1)  # Example for "sampling"
# paramBreaks <- c(0, 40, 80, 120, 160, 200)  # Example for "context_span" and "target_length"

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

safe_scale_minmax <- function(x) {
  if (length(unique(x)) < 2) {
    return(rep(0, length(x)))  # If only 1 unique value, return 0
  } else {
    return(2 * ((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))) - 1)
  }
}

######################################
## 3) Figure 4A Data: dfA_melt
##    (All word-pairs & consecutive words)
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
  # If you want to remove NA but keep 0, do filter(!is.na(value)).
  filter(!is.na(paramValue) & !is.na(value) & value != 0) %>%
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
  ) %>%
  filter(!is.na(gParam))  # Remove any NA values from binning

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
## 4) Updated Figure 4A
######################################
fig4A <- ggplot(dfA_melt, aes(x = gParam, y = value2)) +
  stat_summary(fun = "mean", color = "black", size = 1, aes(group = 1)) +
  stat_summary(fun = "mean", color = "black", size = 2, geom = "line", linetype = "solid", aes(group = 1)) +
  facet_grid(~variable, scales = "free") +
  stat_summary(aes(color = factor(prompt_number), group = factor(prompt_number)),
               fun = "mean", geom = "line", alpha = 0.5) +
  theme_bw() +
  xlab(paramName) +
  ylab("Normalized semantic distance") +
  ggtitle("Average semantic distances between words") +
  theme(
    axis.title    = element_text(size = 14),
    axis.text     = element_text(size = 12),
    strip.text    = element_text(size = 12),
    plot.title    = element_text(size = 20), # Larger and to the left  hjust = -0.2, vjust = -1
    legend.text   = element_text(size = 11),
    legend.title  = element_text(size = 14)
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
  filter(!is.na(paramValue) & !is.na(value)) %>%
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
  ) %>%
  filter(!is.na(gParam))  # Remove NA values from binning

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
  ggtitle("Average semantic distances from the prompt") +
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
    gParam = cut(paramValue, breaks = paramBreaks, include.lowest=TRUE),
    sent   = sentence_number
  ) %>%
  filter(!is.na(gParam))  # Remove NA values from binning

dfB_tangential$gParam <- factor(dfB_tangential$gParam)

# Count number of datapoints per sentence number and gParam
dfB_tangential_count <- dfB_tangential %>%
  group_by(sent, gParam, variable) %>%
  summarize(n = n(), .groups = 'drop') %>%
  mutate(
    line_alpha_value  = scales::rescale(n, to = c(0.2, 1)),   # Transparency for lines
    point_alpha_value = scales::rescale(n, to = c(0.3, 0.5)), # Transparency for dots (avoid complete invisibility)
    line_size_value   = scales::rescale(n, to = c(0.2, 0.5)),   # Line width scaling
    point_size_value  = scales::rescale(n, to = c(0.5, 1))      # Dot size scaling
  )

# Merge the computed alpha and size values back into the main dataframe
dfB_tangential <- dfB_tangential %>%
  left_join(dfB_tangential_count, by = c("sent", "gParam", "variable"))

######################################
## 8) Figure 4C
######################################
# Update Figure 4C with dynamic size for lines and points, and remove alpha from the legend
fig4C <- ggplot(dfB_tangential, aes(x = sent, y = value2, color = gParam, group = gParam)) +
  # Line: Independent Transparency + Size
  stat_summary(fun = "mean", geom = "line", 
               aes(alpha = line_alpha_value, size = line_size_value), na.rm = TRUE) +  
  # Point: Independent Transparency + Size
  stat_summary(fun = "mean", geom = "point", 
               aes(alpha = point_alpha_value, size = point_size_value), na.rm = TRUE) +  
  facet_grid(~variable) +
  theme_bw() +
  xlab("Sentence #") +
  ylab("Normalized semantic distance") +
  scale_color_viridis_d(option = "C", end = 0.8) +
  scale_alpha_continuous(range = c(0.2, 1), guide = "none") +  # Remove alpha from legend
  scale_size_continuous(range = c(0.5, 6)) +  # Ensures independent size control
  labs(color = "Parameter bin") +
  ggtitle("Semantic distance from prompt over sentences") +
  theme(
    axis.title      = element_text(size = 14),
    axis.text       = element_text(size = 12),
    strip.text      = element_text(size = 12),
    legend.position = "top",
    legend.text     = element_text(size = 12),
    legend.title    = element_text(size = 12),
    plot.title      = element_text(size = 16),
    # Add fine-grained grid lines
    panel.grid.major = element_line(size = 0.5, color = "grey75"),  # Make major grid lines slightly visible
    panel.grid.minor = element_line(size = 0.3, color = "grey80"),  # Add fine-grained minor grid lines
    panel.grid.major.x = element_line(size = 0.5, color = "grey75"), # Apply to x-axis
    panel.grid.minor.x = element_line(size = 0.5, color = "grey80"), # Finer x-axis grid
    panel.grid.major.y = element_line(size = 0.5, color = "grey75"), # Apply to y-axis
    panel.grid.minor.y = element_line(size = 0.1, color = "grey80")  # Finer y-axis grid
  ) +
  guides(size = "none")  # Remove size from legend

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


# # For Figure 4A (dfA_melt)
# high_values_A <- dfA_melt %>%
#   filter(value2 > 0.4)

# # For Figure 4B (dfB_melt)
# high_values_B <- dfB_melt %>%
#   filter(value2 > 0.4)

# # For Figure 4C (dfB_tangential)
# high_values_C <- dfB_tangential %>%
#   filter(value2 > 0.4)

# # Combine results into a single summary
# high_values_summary <- list(
#   "Figure 4A" = high_values_A,
#   "Figure 4B" = high_values_B,
#   "Figure 4C" = high_values_C
# )

# # Display results
# high_values_summary