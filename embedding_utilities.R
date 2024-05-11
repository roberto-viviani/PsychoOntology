# define matlab's pdist2
library(Rcpp)
library(dplyr)

# Euclidean distance
cppFunction('NumericMatrix crossdist(NumericMatrix m1, NumericMatrix m2) {

  int nrow1 = m1.nrow();
  int nrow2 = m2.nrow();
  int ncol = m1.ncol();
 
  if (ncol != m2.ncol()) {
    throw std::runtime_error("Incompatible number of columns");
  }
 
  NumericMatrix out(nrow1, nrow2);
 
  for (int r1 = 0; r1 < nrow1; ++r1) {
    for (int r2 = 0; r2 < nrow2; ++r2) {
      double total = 0;
      for (int c = 0; c < ncol; ++c) {
        total += pow(m1(r1, c) - m2(r2, c), 2);
      }
      out(r1, r2) = sqrt(total);
    }
  }
 
  return out;

}')

# cosine distance
cppFunction('NumericMatrix crcosdist(NumericMatrix m1, NumericMatrix m2) {

  int nrow1 = m1.nrow();
  int nrow2 = m2.nrow();
  int ncol = m1.ncol();
 
  if (ncol != m2.ncol()) {
    throw std::runtime_error("Incompatible number of columns");
  }
  
  // length of rows of matrix 1
  NumericMatrix lenmx1(nrow1, 1);
  for (int r = 0; r < nrow1; ++r) {
    double total = 0.0;
    for (int c = 0; c < ncol; ++c) 
      total += pow(m1(r, c), 2);
    lenmx1(r, 0) = sqrt(total);
  }
    
  // length of rows of matrix 2
  NumericMatrix lenmx2(nrow2, 1);
  for (int r = 0; r < nrow2; ++r) {
    double total = 0.0;
    for (int c = 0; c < ncol; ++c) 
      total += pow(m2(r, c), 2);
    lenmx2(r, 0) = sqrt(total);
  }
 
  NumericMatrix out(nrow1, nrow2);
 
  for (int r1 = 0; r1 < nrow1; ++r1) {
    for (int r2 = 0; r2 < nrow2; ++r2) {
      double total = 0.0;
      for (int c = 0; c < ncol; ++c) {
        total += m1(r1, c) * m2(r2, c);
      }
      out(r1, r2) = 1.0 - total / (lenmx1(r1, 0) * lenmx2(r2, 0));
    }
  }
 
  return out;

}')

# Transform the json representation of the embedding into vector
# esize is the the size of the embedding. The json represents a list
decode <- function(eb) {
  # utility to decode json strings
  em <- sub("\\[", "c\\(", eb)
  em <- sub("\\]", "\\)", em)
  c <- eval(str2expression(em))
}

emblength <- function(frm) {
  # utility to 
  l <- length(decode(frm[1, "embedding"]))
  l
}
embmx <- function(frm) {
  # determine size of embedding
  esize <- length(decode(frm[1, "embedding"]))
  print(paste("Embedding of size", esize))

  # decode embeddings  
  mx = matrix(nrow = nrow(frm), ncol = esize, data = 0)
  for (count in 1 : nrow(frm)) 
    mx[count,] <- decode(frm[count, "embedding"])
  mx
}

# provide cosine distance matrix from language model cleanNEO excludes NEO items over #60
loadEmbeddings <- function(embeddings_file, cleanNEO = TRUE) {
  embeddings <- read.csv(embeddings_file)
  
  NEO <- filter(embeddings, scaleID == "NEO") |> select(itemID, type, embedding)
  if (cleanNEO)
    NEO <- NEO[as.integer(row.names(NEO)) < 61,]
  PID <- filter(embeddings, scaleID == "PID") |> select(itemID, type, embedding)
  NEOmx <- embmx(NEO)
  PIDmx <- embmx(PID)

  cdist <- crcosdist(NEOmx, PIDmx)
  retval <- list(cdist = cdist, NEO = NEO, PID = PID)
}

# mapping from PID facets to traits
# some facets are misspelled in the original dataset, correction here
traits <- c(
  "emotional lability" = "NegativeAffect",
  "anxiousness" = "NegativeAffect",
  "anxiety" = "NegativeAffect",
  "separation insecurity" = "NegativeAffect",
  "withdrawal" = "Detachment",
  "intimacy avoidance" = "Detachment",
  "intimicy avoidance" = "Detachment",
  "anhedonia" = "Detachment",
  "manipulativeness" = "Antagonism",
  "deceitfulness" = "Antagonism",
  "grandiosity" = "Antagonism",
  "irresponsibility" = "Disinhibition",
  "impulsivity" = "Disinhibition",
  "distractibility" = "Disinhibition",
  "eccentricity" = "Psychoticism",
  "eccentrictiy" = "Psychoticism",
  "perceptual dysregulation" = "Psychoticism",
  "perceptual dysreg" = "Psychoticism",
  "unusual beliefs" = "Psychoticism",
  "perfectionism" = "Anankastia",
  "rigid perfectionism" = "Anankastia",
  "rigidity" = "Anankastia",
  "orderliness" = "Anankastia"
)
