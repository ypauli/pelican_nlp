# Load required libraries
library(ggplot2)
library(igraph)
library(gridExtra)

# Example words and coordinates
words <- c("day", "eat", "start", "people", "jam", "butter", "bread")
coordinates <- data.frame(
  x = c(1, 2, 3, 5, 3, 4, 4.5),
  y = c(1, 1.5, 3, 2.5, 0.5, 0.8, 0.2),
  label = words
)

# Panel A: Consecutive Words
edges_consecutive <- data.frame(
  x = coordinates$x[-nrow(coordinates)],
  y = coordinates$y[-nrow(coordinates)],
  xend = coordinates$x[-1],
  yend = coordinates$y[-1]
)

plot_a <- ggplot() +
  geom_point(data = coordinates, aes(x, y), color = "blue", size = 3) +
  geom_text(data = coordinates, aes(x, y, label = label), nudge_y = 0.2) +
  geom_segment(data = edges_consecutive, aes(x, y, xend = xend, yend = yend), 
               arrow = arrow(length = unit(0.2, "cm")), color = "orange") +
  ggtitle("Panel A: Consecutive Words") +
  theme_minimal()

# Panel B: All Word Pairs
graph <- graph_from_data_frame(d = expand.grid(words, words), directed = FALSE)
edge_list <- as_edgelist(graph)  # Replacing deprecated `get.edgelist()`
edges_all_pairs <- data.frame(
  x = coordinates$x[match(edge_list[, 1], words)],
  y = coordinates$y[match(edge_list[, 1], words)],
  xend = coordinates$x[match(edge_list[, 2], words)],
  yend = coordinates$y[match(edge_list[, 2], words)]
)

plot_b <- ggplot() +
  geom_point(data = coordinates, aes(x, y), color = "blue", size = 3) +
  geom_text(data = coordinates, aes(x, y, label = label), nudge_y = 0.2) +
  geom_segment(data = edges_all_pairs, aes(x = x, y = y, xend = xend, yend = yend), 
               alpha = 0.5, color = "grey") +
  ggtitle("Panel B: All Word Pairs") +
  theme_minimal()

# Combine both plots side by side and print
plots <- gridExtra::grid.arrange(plot_a, plot_b, nrow = 2)
print(plots)