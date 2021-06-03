library(dplyr)
library(reshape2)
library(ggplot2)

setwd("CS/2021/B/IML/Hackathon/")

df <- read.delim("movies_dataset.csv", sep=",")

df_nums <- df %>% select(id, budget, vote_average, vote_count, runtime, revenue, original_language)

summary(df)
mm <- melt(df_nums)  
head(mm)

gg <- ggplot(mm, aes(x=value))+
  geom_histogram()+
  facet_wrap(~variable, scales = "free")+
  labs("Counts values per variable")+
  theme_bw()
  
gg
ggsave(plot = gg, "Counts.png", device = 'png')
  

corrplot::corrplot(corr = df_nums, is.corr = F)
