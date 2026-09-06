library(scDblFinder)
library(SingleCellExperiment)
# 1) 转成 SCE
sce <- as.SingleCellExperiment(experiment.aggregate)
# 2) 按样本分开跑（强烈建议）
sce <- scDblFinder(sce, samples = sce$patients)  # sample字段名按你自己的改
# 3) 把结果写回 Seurat
experiment.aggregate$doublet_class <- colData(sce)$scDblFinder.class    # "doublet"/"singlet"
experiment.aggregate$doublet_score <- colData(sce)$scDblFinder.score
FeaturePlot(experiment.aggregate, features=c("doublet_score"), order=TRUE)
# 4) 剔除双细胞
experiment.aggregate<- subset(experiment.aggregate, subset = doublet_class == "singlet")






# 加载必要的包
library(decontX)
library(ggplot2)
##单个分开，用来做RNA污染
sce_list <- SplitObject(experiment.aggregate, split.by = "patients")
# 1. 对sce_list中所有对象进行去污染处理
for (i in seq_along(sce_list)) {
  # 获取counts数据
  counts <- GetAssayData(object = sce_list[[i]], slot = "counts")
  # 运行decontX去污染
  decontX_results <- decontX(counts)
  # 添加污染分数到metadata
  sce_list[[i]]$Contamination <- decontX_results$contamination
  # 过滤高污染的细胞（污染分数 < 0.2）
  sce_list[[i]] <- sce_list[[i]][, sce_list[[i]]$Contamination < 0.2]
  cat("处理完成对象", i, "，剩余细胞数：", ncol(sce_list[[i]]), "\n")
}
# 2. 合并所有处理后的对象
# 方法1：使用Reduce和merge
rawdata <- Reduce(function(x, y) merge(x, y), sce_list)
experiment.aggregate=rawdata
experiment.aggregate=JoinLayers(experiment.aggregate)
save(experiment.aggregate,file=paste0("1-34raw.RData"))