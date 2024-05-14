library(tidyverse)
library(irr)

# Specify the path to your CSV file
file <- "csv_file_kripp_alpha.csv"


# Read the CSV file
data <- read.csv(file)

## ASYMMETRY
f1_col <- c("Anno1Feat1", "Anno2Feat1", "Anno3Feat1", "Anno4Feat1") # Columns with feature 1 (asymmetry)
asym_data <- data[, which(names(data) %in% f1_col)] # Dataframe with only asymmetry columns
asym_matrix <- as.matrix(asym_data) # Change dataframe to matrix
asym_t_matrix <- t(asym_matrix) # Transpose matrix
asym_k_a <- kripp.alpha(asym_t_matrix, "ordinal") # Calculating Krippendorffs Alpha  

print(asym_k_a) # Prints Krippendorffs Alpha for ASYMMETRY

## RESULT for ASYMMETRY
## Subjects = 108, Raters = 4, alpha = 0.568

## COLOR
f2_col <- c("Anno1Feat2", "Anno2Feat2", "Anno3Feat2", "Anno4Feat2") # Columns with feature 2 (color)
color_data <- data[, which(names(data) %in% f2_col)]
color_matrix <- as.matrix(color_data) # Change dataframe to matrix
color_t_matrix <- t(color_matrix) # Transpose matrix
color_k_a <- kripp.alpha(color_t_matrix, "ordinal") # Calculating Krippendorffs Alpha

print(color_k_a) # Prints Krippendorffs Alpha for COLOR

## RESULT for COLOR
## Subjects = 108, Raters = 4, alpha = 0.728

## DOTS AN GLOBULES
f3_col <- c("Anno1Feat3", "Anno2Feat3", "Anno3Feat3", "Anno4Feat3") # Columns with feature 3 (dots and globules)
dots_glob_data <- data[, which(names(data) %in% f3_col)]
dots_glob_matrix <- as.matrix(dots_glob_data) # Change dataframe to matrix
dots_glob_t_matrix <- t(dots_glob_matrix) # Transpose matrix
dots_glob_k_a <- kripp.alpha(dots_glob_t_matrix, "ordinal") # Calculating Krippendorffs Alpha

print(dots_glob_k_a) # Prints Krippendorffs Alpha for DOTS AND GLOBULES

## RESULT for DOTS AND GLOBULES
## Subjects = 108, Raters = 4, alpha = 0.606

