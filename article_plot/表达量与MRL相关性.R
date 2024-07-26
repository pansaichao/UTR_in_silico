rm(list = ls())
library(ggplot2) # 加载ggplot2包
library(dplyr) # 加载dplyr包
library(ggpmisc) # 加载ggpmisc包
library(ggsci) # 调色板
library(gridExtra)
library(scales)
library(ggrepel)
library(gridExtra)
library(grid)
library(tidyr)

# 读取数据
flow_data1 <- read.table("./UTR_data/RHTR0032-GFP更正_论文.csv", sep = ",", head = TRUE, encoding = "utf-8")

flow_data1_subset <- flow_data1[flow_data1$name != 'UTR_80', ]
flow_data1_subset <- flow_data1

correlation <- cor(flow_data1_subset$live.Mean.FITC.A_24H, flow_data1_subset$pred_frame_pool)
cor_value <- correlation[1]
p_value <- cor.test(flow_data1_subset$live.Mean.FITC.A_24H, flow_data1_subset$pred_frame_pool)$p.value

# 自定义颜色调色板
custom_palette <- c("#4c72b0", "#dd8452", "#55a868")

mrl_fitc_mean_24h <- ggplot(flow_data1_subset,
                            aes(x = pred_frame_pool, y = live.Mean.FITC.A_24H, colour = group)) +
  geom_point(size = 3) +
  geom_text_repel(aes(label = name), size = 3) +
  labs(x = 'Predicted MRL', y = '24-hour average fluorescence intensity of GFP', title = paste("Correlation:", round(cor_value, 2), "  ", "p-value:", format(p_value, scientific = TRUE, digits = 3))) +
  theme_bw() +
  geom_smooth(color = "skyblue", formula = y ~ x, fill = "skyblue", method = "lm") +
  scale_color_manual(values = custom_palette) +
  theme(
    panel.grid.major = element_blank(), # 去掉主要网格线
    panel.grid.minor = element_blank(), # 去掉次要网格线
    legend.title = element_blank(), # 去掉标签的标题
    axis.title.x = element_text(size = 14), # 修改x轴标题的字体大小
    axis.title.y = element_text(size = 14) # 修改y轴标题的字体大小
  )

# 保存为TIFF格式
ggsave("./result_3/各组utr的MRL预测值与24h平均荧光强度相关性.tiff", plot = mrl_fitc_mean_24h, device = "tiff", width = 15, height = 9, dpi = 600, compression = "lzw")


# **********************************平均荧光强度48h*****************************************

correlation <- cor(flow_data1_subset$live.Mean.FITC.A_48H, flow_data1_subset$pred_frame_pool)
cor_value <- correlation[1]
p_value <- cor.test(flow_data1_subset$live.Mean.FITC.A_48H, flow_data1_subset$pred_frame_pool)$p.value

mrl_fitc_mean_48h <- ggplot(flow_data1_subset,
                            aes(x = pred_frame_pool, y = live.Mean.FITC.A_48H, colour = group)) +
  geom_point(size = 3) +
  geom_text_repel(aes(label = name), size = 3) +
  labs(x = 'Predicted MRL', y = '48-hour average fluorescence intensity of GFP', title = paste("Correlation:", round(cor_value, 2), "  ", "p-value:", format(p_value, scientific = TRUE, digits = 3))) +
  theme_bw() +
  geom_smooth(color = "skyblue", formula = y ~ x, fill = "skyblue", method = "lm") +
  scale_color_manual(values = custom_palette) +
  theme(
    panel.grid.major = element_blank(), # 去掉主要网格线
    panel.grid.minor = element_blank(), # 去掉次要网格线
    legend.title = element_blank(), # 去掉标签的标题
    axis.title.x = element_text(size = 14), # 修改x轴标题的字体大小
    axis.title.y = element_text(size = 14) # 修改y轴标题的字体大小
  )

# 保存为TIFF格式
ggsave("./result_3/各组utr的MRL预测值与48h平均荧光强度相关性.tiff", plot = mrl_fitc_mean_48h, device = "tiff", width = 15, height = 9, dpi = 600, compression = "lzw")


# **********************************分组计算荧光强度倍数24h*****************************************


# 计算每个组的平均值
group_means <- flow_data1 %>%
  group_by(group) %>%
  summarise(mean_value = mean(live.Mean.FITC.A_24H, na.rm = TRUE))

print(group_means)

# 提取 in silico 组、literature 组和 NGS 组的平均值
mean_in_silico <- group_means %>% filter(group == "In silico") %>% pull(mean_value)
mean_literature <- group_means %>% filter(group == "Literature") %>% pull(mean_value)
mean_ngs <- group_means %>% filter(group == "NGS") %>% pull(mean_value)

# 计算 in silico 组平均值是其他两组的几倍
ratio_in_silico_literature <- mean_in_silico / mean_literature
ratio_in_silico_ngs <- mean_in_silico / mean_ngs

print(paste("in silico 组平均值是 literature 组平均值的", round(ratio_in_silico_literature, 2), "倍"))
print(paste("in silico 组平均值是 NGS 组平均值的", round(ratio_in_silico_ngs, 2), "倍"))

# **********************************分组计算荧光强度倍数48h*****************************************


# 计算每个组的平均值
group_means <- flow_data1 %>%
  group_by(group) %>%
  summarise(mean_value = mean(live.Mean.FITC.A_48H, na.rm = TRUE))

print(group_means)

# 提取 in silico 组、literature 组和 NGS 组的平均值
mean_in_silico <- group_means %>% filter(group == "In silico") %>% pull(mean_value)
mean_literature <- group_means %>% filter(group == "Literature") %>% pull(mean_value)
mean_ngs <- group_means %>% filter(group == "NGS") %>% pull(mean_value)

# 计算 in silico 组平均值是其他两组的几倍
ratio_in_silico_literature <- mean_in_silico / mean_literature
ratio_in_silico_ngs <- mean_in_silico / mean_ngs

print(paste("in silico 组平均值是 literature 组平均值的", round(ratio_in_silico_literature, 2), "倍"))
print(paste("in silico 组平均值是 NGS 组平均值的", round(ratio_in_silico_ngs, 2), "倍"))
