library(tidyverse)
library(irr)

# Load in manual scores CSV file
manual_scores <- read.csv('csv_file_kripp_alpha.csv')

# Load in CSV file with generated features scores
# CHANGE THIS FILE PATH AS NEEDED 
generated_scores <- read.csv('features.csv')

# Merge the two dataframes
data <- cbind(manual_scores, generated_scores)

# Drop 2nd image_id column
data = subset(data, select = c(0:17,19:21))

# Change the asymmetry cols values to better match our manual annnotation scores
data$asymmetry[data$asymmetry >= 0 & data$asymmetry < 0.33] <- 0
data$asymmetry[data$asymmetry >= 0.33 & data$asymmetry < 0.66] <- 1
data$asymmetry[data$asymmetry >= 0.66 & data$asymmetry <= 1] <- 2

# Calculate Krippendorffs Alpha

## ASYMMETRY
f1_col <- c("Anno1Feat1", "Anno2Feat1", "Anno3Feat1", "Anno4Feat1", "asymmetry") # Columns with feature 1 (asymmetry)
asym_data <- data[, which(names(data) %in% f1_col)] # Dataframe with only asymmetry columns
asym_matrix <- as.matrix(asym_data) # Change dataframe to matrix
asym_t_matrix <- t(asym_matrix) # Transpose matrix
asym_k_a <- kripp.alpha(asym_t_matrix, "ordinal") # Calculating Krippendorffs Alpha  

print(asym_k_a) # Prints Krippendorffs Alpha for ASYMMETRY

## RESULT for ASYMMETRY
## Subjects = 106, Raters = 5, alpha = 0.265

## COLOR
f2_col <- c("Anno1Feat2", "Anno2Feat2", "Anno3Feat2", "Anno4Feat2", "colour") # Columns with feature 2 (colour)
color_data <- data[, which(names(data) %in% f2_col)]
color_matrix <- as.matrix(color_data) # Change dataframe to matrix
color_t_matrix <- t(color_matrix) # Transpose matrix
color_k_a <- kripp.alpha(color_t_matrix, "ordinal") # Calculating Krippendorffs Alpha

print(color_k_a) # Prints Krippendorffs Alpha for COLOR

## RESULT for COLOUR
## Subjects = 106, Raters = 5, alpha = 0.0771 

## DOTS AN GLOBULES
f3_col <- c("Anno1Feat3", "Anno2Feat3", "Anno3Feat3", "Anno4Feat3", "dots") # Columns with feature 3 (dots adn globules)
dots_glob_data <- data[, which(names(data) %in% f3_col)]
dots_glob_matrix <- as.matrix(dots_glob_data) # Change dataframe to matrix
dots_glob_t_matrix <- t(dots_glob_matrix) # Transpose matrix
dots_glob_k_a <- kripp.alpha(dots_glob_t_matrix, "ordinal") # Calculating Krippendorffs Alpha

print(dots_glob_k_a) # Prints Krippendorffs Alpha for DOTS AND GLOBULES

## RESULT for DOTS AND GLOBULES
## Subjects = 106, Raters = 5, alpha = 0.197