library(tidyverse)
library(CytoTRACE2)
library(Seurat)
library(paletteer)
library(BiocParallel)
library("ggunchull")
library(tidydr)
cytotrace2_res <- cytotrace2(sce,#seurat对象
                             is_seurat =TRUE,
                             slot_type ="counts",#counts和data都可以
                             species ='human')#物种要选择，默认是小鼠


# plotting-一次性生成多个图，然后储存在一个list，用$查看即可
annotation <- data.frame(phenotype = sce@meta.data$seurat_clusters) %>%
  set_rownames(., colnames(sce))

# plotting-一次性生成多个图，然后储存在一个list，用$查看即可
plots <- plotData(cytotrace2_result = cytotrace2_res,annotation=annotation,
                  is_seurat =TRUE)
plots$CytoTRACE2_UMAP+NoAxes()+
  theme(aspect.ratio=1) +
  theme_dr()+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
FeaturePlot(sce,
        features ="CytoTRACE2_Score",
        pt.size = 0.3,
        label = F,raster=FALSE,split.by = "type")
DimPlot(object =cytotrace2_res, pt.size=0.5,
        reduction = "umap",label = T) +
  ggsci::scale_color_d3()


plots$CytoTRACE2_UMAP

plots$CytoTRACE2_Potency_UMAP

plots$CytoTRACE2_Relative_UMAP

plots$Phenotype_UMAP

plots$CytoTRACE2_Boxplot_byPheno

saveRDS(cytotrace2_res,file="ep3.rds")

library(ggplot2)

df <- data.frame(
  UMAP_1 = Embeddings(sce, "umap")[,1],
  UMAP_2 = Embeddings(sce, "umap")[,2],
  CytoTRACE2 = cytotrace2_res$CytoTRACE2_Score,
  type = cytotrace2_res$type
)

ggplot(df, aes(UMAP_1, UMAP_2, color = CytoTRACE2)) +
  geom_point(size = 0.3) +
  facet_wrap(~type) +
  theme_classic()
