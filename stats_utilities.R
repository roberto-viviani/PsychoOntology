# Utilities to estimate stats

# utility to estimate pearson correlations repeatedly
pearscorr <- function(data, yvar, xvars) {
  stats <- data.frame()
  for (v in xvars) {
    cs <- cor.test(pull(data,yvar), pull(data,v), 
                   method = "pearson", alternative = "two.sided",
                   na.action = "na.omit")
    stats[v,"r"] <- cs$estimate
    stats[v,"t"] <- cs$statistic
    stats[v,"p"] <- cs$p.value
    stats[v, "pcorr"] <- ifelse(cs$p.value < 0.001 / length(xvars) / 5, "***", 
                                ifelse(cs$p.value < 0.01 / length(xvars) / 5, "**", 
                                       ifelse(cs$p.value < 0.05 / length(xvars) / 5, 
                                              "*", "-")))
  }
  stats
}

# utility to estimate anova for distances
library(lme4)
anovadist <- function(plotdata) {
  test <- anova(lmer(cosdist ~ 1 + (1 | itemNEO) + (1 | itemPID), data=plotdata),
                lmer(cosdist ~ traitPID + (1 | itemNEO) + (1 | itemPID), data=plotdata))
  print(paste("Chi square:", test$Chisq[2], ", df", test$Df[2], ", p =", 
              test$`Pr(>Chisq)`[2]))
}
