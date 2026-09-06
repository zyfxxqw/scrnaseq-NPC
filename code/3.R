library(scDblFinder)
library(SingleCellExperiment)

sce <- as.SingleCellExperiment(experiment.aggregate)
sce <- scDblFinder(sce, samples = sce$patients) 
experiment.aggregate$doublet_class <- colData(sce)$scDblFinder.class    # "doublet"/"singlet"
experiment.aggregate$doublet_score <- colData(sce)$scDblFinder.score
FeaturePlot(experiment.aggregate, features=c("doublet_score"), order=TRUE)
experiment.aggregate<- subset(experiment.aggregate, subset = doublet_class == "singlet")



# decontX
library(decontX)
library(ggplot2)

sce_list <- SplitObject(experiment.aggregate, split.by = "patients")
for (i in seq_along(sce_list)) {

  counts <- GetAssayData(object = sce_list[[i]], slot = "counts")

  decontX_results <- decontX(counts)

  sce_list[[i]]$Contamination <- decontX_results$contamination

  sce_list[[i]] <- sce_list[[i]][, sce_list[[i]]$Contamination < 0.2]
  cat("处理完成对象", i, "，剩余细胞数：", ncol(sce_list[[i]]), "\n")
}
rawdata <- Reduce(function(x, y) merge(x, y), sce_list)
experiment.aggregate=rawdata
experiment.aggregate=JoinLayers(experiment.aggregate)
save(experiment.aggregate,file=paste0("1-34raw.RData"))
