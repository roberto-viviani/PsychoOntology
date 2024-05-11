# Plot utilities

# utility function for base plot
baseplot <- function(plotdata, facetflag = FALSE) {
  p <- ggplot(plotdata, aes(cosdist, corrs))
  if (facetflag) {
    p <- p + geom_point(aes(color = typePID), alpha=0.5, shape=16, size=2.4) +
      scale_colour_viridis_d(name = "PID facet", option = "H", direction = -1)
  }
  else {
    p <- p +  geom_point(aes(color = traitPID), alpha=0.5, shape=16, size=2.4) +
      scale_colour_viridis_d(name = "PID trait", option = "H", direction = -1,
                             labels = c("Anancasticity", "Antagonism", "Detachment",
                                        "Disinhibition", "Negative affect",
                                        "Psychoticism"))
  }
  p <- p + labs(title = NULL) +
    xlab('semantic distance (cosine)') + ylab('responses correlation') + 
    theme_classic() + theme(legend.position = "left")
  p
}

# utility function to add line to plot
addneoline <- function(p, plotdata, neoscale) {
  ldata <- layer_data(p)
  traitcolours <- data.frame(colour = ldata$colour, 
                             red = col2rgb(ldata$colour)[1,],
                             green = col2rgb(ldata$colour)[2,],
                             blue = col2rgb(ldata$colour)[3,],
                             group = ldata$group, 
                             traitPID = plotdata$traitPID) |>
    group_by(traitPID) |> 
    summarise(red = mean(red), green = mean(green), blue = mean(blue))
  color <- rgb(traitcolours |> filter(traitPID == neoscale) |> 
                 select(red, green, blue),
               maxColorValue = 255, alpha = 0.7*255)
  p <- p + geom_smooth(data = filter(plotdata, traitPID == neoscale), 
                       method = "lm", se = FALSE, color = color)
  p
}

# utility to do a bar plot of correlations
barplf <- function(corrs, ptitle) {
  p <- ggplot(corrs, aes(x = row.names(corrs), y = r, fill = row.names(corrs))) + 
    geom_col() + 
    scale_fill_viridis_d(option = "H", direction = -1,
                         labels = c("Anancasticity", "Antagonism", "Detachment",
                                    "Disinhibition", "Negative affect",
                                    "Psychoticism")) + 
    labs(title = ptitle, fill = "PID trait") + 
    ylab("correlation responses") + xlab(NULL) +
    theme_classic() + 
    theme(panel.grid.major.y = element_line(), 
          axis.text.x = element_blank(), axis.ticks.x = element_blank(),
          text = element_text(size = 14), aspect.ratio = 2)
  p
}

# utility to do a boxplot of semantic distances
bxplf <- function(plotdata) {
  labls = c("Anancasticity", "Antagonism", "Detachment",
            "Disinhibition", "Negative affect",
            "Psychoticism")
  p <- ggplot(plotdata, aes(x = traitPID, y = cosdist)) + 
    geom_boxplot(aes(fill = traitPID), alpha = 0.6) + 
    geom_jitter(aes(color = traitPID), alpha = 0.4, width = 0.1) + 
    scale_fill_viridis_d(option = "H", direction = -1, labels = labls) + 
    scale_color_viridis_d(option = "H", direction = -1, labels = labls) + 
    ylab("semantic distance (cosine)") + xlab(NULL) +
    theme_classic() + labs(fill = "PID trait", color = "PID trait") +
    theme(panel.grid.major.y = element_line(), 
          axis.text.x = element_blank(), axis.ticks.x = element_blank(),
          text = element_text(size = 14), aspect.ratio = 2)
  p
}
