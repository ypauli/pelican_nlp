# -----------------------------------------------------------
#  Example Derailment Plot (like Figures 4A or 4B)
# -----------------------------------------------------------
# Assumes your data frame is something like df_temp1_agg or df_back1_agg 
# (i.e. an aggregated dataset). You need at least:
#   x-axis factor  : gTemp (or gBackSpan)
#   y-axis numeric : value2
#   grouping vars  : Prompt (for color), Model (for shape/linetype)
#   facet variable : variable (e.g. "Word mover's distance", etc.)
#   possibly 'stat' if you only want subset = "Average"

# Import necessary libraries:
library(ggplot2)
library(dplyr)

# Load the data from the CSV files and combine them into one data frame
fp <- list.files(path = "/home/ubuntu/emilia/PELICAN/pelican/simulation/simu_analysis/data/", 
                 pattern = "temperature-[0-3].csv", full.names = TRUE)

df_list <- lapply(seq_along(fp), function(i) {
  df <- read.csv(fp[i])
  df$Prompt <- as.factor(i - 1)  # Add the 'Prompt' column
  return(df)
})

df_combined <- bind_rows(df_list)

p <- ggplot(
  df_combined,
  # subset(file, stat == "Average"),  # or any subset you want
  aes(x = temperature, y = semantic_distance)
) +
  # 1) Black points for overall mean by bin:
  stat_summary(fun.y = "mean", color = "black", size = 1, aes(group = 1)) +

  # 2) Black line for overall mean by bin:
  stat_summary(
    fun.y = "mean", color = "black", size = 2,
    geom = "line", linetype = "solid", aes(group = 1)
  ) +

  # 3) Per-prompt colored lines (alpha=0.5 for subtlety):
  stat_summary(aes(color = Prompt, group = Prompt), 
               fun.y = "mean", geom = "line", alpha = 0.5) +

  # 4) Per-model shapes and dashed lines (grey):
  stat_summary(aes(shape = Model, group = Model), 
               fun.y = "mean", color = "grey50") +
  stat_summary(
    aes(linetype = Model, group = Model),
    fun.y = "mean", geom = "line", size = 1, color = "grey50"
  ) +

  # 5) Manually control shape and linetype legends:
  scale_shape_manual(values = c(0, 1, 2)) +
  scale_linetype_manual(values = c(2, 3, 4)) +

  # 6) Facet by distance measure type:
  facet_grid(~ semantic_distance, scales = "free") +

  # 7) Set y-limits, labels, etc.
  ylim(-1, 1) +
  xlab("Temperature (or Memory Span)") +
  ylab("Normalized semantic distance") +

  # 8) Theme and text sizes:
  theme_bw() +
  theme(
    axis.title  = element_text(size = 14),
    axis.text   = element_text(size = 12),
    strip.text  = element_text(size = 12),
    plot.title  = element_text(size = 16),
    legend.text = element_text(size = 11),
    legend.title= element_text(size = 14),
    legend.box.margin = margin(r = 0, l = 0)
  ) +



  # 10) Adjust legend layout (optional):
  guides(
    linetype = guide_legend(title = "Embedding Model", nrow = 1),
    shape    = guide_legend(title = "Embedding Model", nrow = 1),
    color    = guide_legend(nrow = 3)  # or as needed
  ) +

  ggtitle("Semantic Distances Between Words (or Sentences)")

print(p)