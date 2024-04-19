# This is a utility to prepare scales to be embedded starting from an excel
# spreadsheet, like Scales.xls. It loads the scales data from this excel
# file and saves a scales.csv file in the main directory.
#
# The embedding code expects a csv file with the columns item_de and item_en
# containing the scale items in German and English, respectively, or a column
# item if there is only one language.´
#
# Roberto Viviani, Institute of Psychology, University of Innsbruck, 2024

library(readxl)
library(dplyr)

# Check that the output file does not exist, and if it does, stop
if (file.exists(".\\scales.csv")) {
  stop("The file scales.csv already exists. Please delete it before running this script.")
}

# Load the scales data from the excel file .\Scales\Scales.xlsx
scales <- readxl::read_excel(".\\Scales\\Scales.xlsx")

# If there is a columns named 'polarity', drop it as responses should
# already have been recoded according to the polarity
if ("polarity" %in% colnames(scales)) {
  scales <- select(scales, -polarity)
}

# Here the code to select the scales that are going to be used
scales <- filter(scales, scaleID == "NEO" | scaleID == "PID" | scaleID == "ADS")

# Save the scales data to a csv file
write.csv(scales, file = ".\\scales.csv", row.names = FALSE)