# Utilities to estimate stats

# utility to estimate pearson correlations repeatedly
pearscorr_ext <- function(data, yvar, xvars) {
  stats <- data.frame()
  for (v in xvars) {
    cs <- cor.test(pull(data,yvar), pull(data,v), 
                   method = "pearson", alternative = "two.sided",
                   na.action = "na.omit")
    stats[v,"r"] <- cs$estimate
    stats[v,"lci"] <- cs$conf.int[1]
    stats[v,"uci"] <- cs$conf.int[2]
    stats[v,"t"] <- cs$statistic
    stats[v,"p"] <- cs$p.value
    stats[v, "pcorr"] <- ifelse(cs$p.value < 0.001 / length(xvars) / 5, "***", 
                                ifelse(cs$p.value < 0.01 / length(xvars) / 5, "**", 
                                       ifelse(cs$p.value < 0.05 / length(xvars) / 5, 
                                              "*", "-")))
  }
  stats
}

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
library(forcats)
library(lmerTest)
anovadist <- function(plotdata) {
  # omnibus test
  test <- anova(lmer(cosdist ~ 1 + (1 | itemNEO) + (1 | itemPID), data=plotdata),
                lmer(cosdist ~ traitPID + (1 | itemNEO) + (1 | itemPID), data=plotdata))
  print(paste("Chi square:", test$Chisq[2], ", df", test$Df[2], ", p =", 
              test$`Pr(>Chisq)`[2]))
  
  # test relative to most distant
  tmp <- lmer(cosdist ~ -1 + traitPID + (1 | itemNEO) + (1 | itemPID), data=plotdata)
  mostdist <- which.max(fixef(tmp))
  mostdist <- sub("traitPID", "", names(mostdist))
  print(c("PID trait with largest distance: ", mostdist))
  traits <- levels(factor(plotdata$traitPID))
  for (i in 1 : length(traits)) {
    if (traits[i] != mostdist) 
      plotdata$traitPID <- fct_relevel(plotdata$traitPID, traits[i], after = Inf)
  }
  print(summary(lmerTest::lmer(cosdist ~ traitPID + (1 | itemNEO) + (1 | itemPID), data=plotdata)))
  
}
