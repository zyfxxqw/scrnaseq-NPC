sce <- readRDS("Myeloid.rds")
obj <- readRDS("ep.rds")
experiment.aggregate <- readRDS("1-33raw.rds")
DefaultAssay(obj)
DefaultAssay(obj) <- "RNA"

# 提取子矩阵重新聚类分析
cells_sub <- subset(experiment.aggregate@meta.data, 
                    new_celltype %in% c("Fibroblasts"))#如果是celltype或者分组重聚类则记得加双引号如celltype %in% c("t cells"))
scRNA_sub <- subset(experiment.aggregate, 
                    cells=row.names(cells_sub))
#剔除某簇
experiment.aggregate<- subset(experiment.aggregate,
                              idents = unique(Idents(experiment.aggregate))[!unique(Idents(experiment.aggregate)) %in% c(20,16)]
)
#在髓系子集里把 cluster16 去掉
sce <- subset(sce, idents =c(""), invert = TRUE)
DimPlot(object = scRNA_sub, pt.size=0.5,label = T)

library(harmony)
sce=scRNA_sub
sce <- NormalizeData(sce, normalization.method = "LogNormalize", scale.factor = 1e4)
sce <- FindVariableFeatures(sce, selection.method = 'vst', nfeatures =2000)
sce <- ScaleData(sce,features = VariableFeatures(sce),
                 vars.to.regress = c("nCount_RNA","percent.mt"))
sce <- RunPCA(sce, features = VariableFeatures(object = sce),npcs = 30)
sce <- RunHarmony(sce, group.by.vars = "patients",dims.use = 1:15,theta=1)
sce <- FindNeighbors(sce, dims = 1:15,reduction = "harmony")
sce <- FindClusters(sce, resolution = 1)
sce <- RunUMAP(sce, dims = 1:15,reduction = "harmony", n.neighbors = 50, min.dist = 0.5)
DimPlot(sce, reduction = 'umap',label = T, pt.size=0.3,raster=T,group.by = "seurat_clusters")
DimPlot(sce, reduction = 'umap',label = T, pt.size=0.3,split.by = "type")
#细胞分群并写入meta
sce<- RenameIdents(
  object = sce,
  "19" = "Malignant_cells","2" = "Malignant_cells","18"="Malignant_cells","5" = "Malignant_cells","1"="Malignant_cells",
  "2" = "Malignant_cells","10" = "Malignant_cells","2"="Malignant_cells","8"="Malignant_cells","17"="Malignant_cells",
  "6"="con","7"="con","14"="con","0"="con","12"="con","12"="con","4"="con","3"="con","11"="con","16"="con","15"="con",
  "9"="con","13"="con")
DimPlot(object =sce, pt.size=0.5,
        reduction = "umap",label = T,group.by = "meta_program_max") +
  ggsci::scale_color_d3()
sce@meta.data$meta_program_max <- Idents(sce)
Idents(sce) <- sce@meta.data$meta_program_max
saveRDS(obj,file="ep.rds")
saveRDS(experiment.aggregate,file="raw1-31.rds")
DimPlot(obj, reduction = 'umap',label = T,group.by = "seurat_clusters")
ap <- FeaturePlot(sce,features=c("MKI67"),ncol = 1,split.by = "patients", combine = FALSE)
wrap_plots(p, ncol = 5)
FeaturePlot(sce,features=c("CLEC9A","VCAN","SPP1","FOLR2","CD1C","C1QA","APOE","LAMP3","S100A8","CSF3R","CLEC4C"),ncol = 5)
FeaturePlot(sce,features=c("CD8A","CXCL13","CTLA4","CD3D","CD4","PDCD1","FOXP3","IL7R","PRF1","CCL5","TIGIT","LAG3","NKG7","GZMB","HAVCR2"),ncol = 5)
FeaturePlot(sce,features=c("EPCAM","KRT19","LAMA1","ITGA2","COL3A1","CD3D","MKI67"),ncol = 3)
FeaturePlot(sce,features=c("HLA-DRA","RGS13","FGR","TXNIP"))
FeaturePlot(sce,features=c("BCL6","MZB1","MS4A1","AICDA","IGHD","IGHG1","FCRL4","CD27","TNFRSF13B","ISG15"),ncol = 3)
FeaturePlot(sce, features = c("PTPRC","COL1A1","DCN","LST1","CD3D","NKG7"))
FeaturePlot(obj, features = c("LGALS9"),split.by = "type")

DotPlot(
  sce,
  features = c("CD8A","CXCL13","CTLA4","CD3D","CD4","PDCD1","FOXP3","IL7R","PRF1","CCL5","TIGIT","LAG3","NKG7","GZMB","HAVCR2"),
  pt.size = 0.1,
  ncol = 3
)
#取出一个亚群重聚类后，要剔除亚群中的一簇并且，返回原seurat对象也剔除此簇
cells_drop <- WhichCells(sce, idents =c(""))
experiment.aggregate <- subset(experiment.aggregate, cells = setdiff(Cells(experiment.aggregate), cells_drop))

#取出一个亚群重聚类后，要剔除亚群中的一簇并且，返回原seurat原seurat对象也剔除此簇
cells_drop <- WhichCells(sce, idents =c("7","2","1","0","11","6","14","17"))
experiment.aggregate <- subset(experiment.aggregate, cells = setdiff(Cells(experiment.aggregate), cells_drop))

#在髓系子集里把 cluster16 去掉
myeloid_clean <- subset(myeloid, idents = 16, invert = TRUE)


# 提取子矩阵重新聚类分析
cells_sub <- subset(sce@meta.data, 
                    type%in% c("CA"))#如果是celltype或者分组重聚类则记得加双引号如celltype %in% c("t cells"))
scRNA_sub <- subset(sce, 
                    cells=row.names(cells_sub))

sce=scRNA_sub
sce <- NormalizeData(sce, normalization.method = "LogNormalize", scale.factor = 1e4)
sce <- FindVariableFeatures(sce, selection.method = 'vst', nfeatures =2000)
sce <- ScaleData(sce,features = VariableFeatures(sce),
                 vars.to.regress = c("nCount_RNA","percent.mt"))
sce <- RunPCA(sce, features = VariableFeatures(object = sce),npcs = 30)

sce <- FindNeighbors(sce, dims = 1:15)
sce <- FindClusters(sce, resolution = 1)
sce <- RunUMAP(sce, dims = 1:15, n.neighbors = 50, min.dist = 0.5)
DimPlot(sce, reduction = 'umap',label = T, pt.size=0.3,raster=F,group.by = "patients")
DimPlot(sce, reduction = 'umap',label = T, pt.size=0.3,raster=T,split.by = "type")

FeaturePlot(obj,features=c("PVR"),ncol = 1, combine = FALSE)


library(reticulate)
# 替换为你刚才找到的路径
use_python("/home/zyf/miniconda3/envs/jupyter/bin/python3", required = TRUE)
#更好用
library(scCustomize)
#install.packages("scCustomize")
# 基础转换命令
as.anndata(
  x = experiment.aggregate, 
  file_path = "./", 
  file_name = "Alladata.h5ad",
  assay = "RNA",               # 指定要转换的 Assay
  main_layer = "data",          # AnnData 的 X 层（通常放归一化后的 data）
  other_layers = "counts",      # 其他层（通常放原始 counts）
  transfer_dimreduc = TRUE      # 是否转换 PCA/UMAP 等降维信息
)

###################################################################################################################
library(Seurat)
library(presto)

# Idents(obj) 需要先设好
Idents(sce) <- obj$celltype   # 或 seurat_clusters

markers <- wilcoxauc(
  X = sce,
  group_by = "seurat_clusters"       # 也可以写 seurat_clusters
)

marker_cut <- markers %>%
  filter(
    padj < 0.05,
    logFC > 0.25,
    pct_in > 0.10
  ) %>%
  arrange(group, padj, desc(logFC))

write.table(marker_cut,
            file=paste0("Myeloidmarker.txt"),
            sep="\t",quote = F,row.names = F)


sce.rename<- RenameIdents(
  object = sce,
"0"  = "Cytotoxic_CD8_T",
"1"  = "Naive_Memory_T",
"2"  = "Treg",
"3"  = "Tfh_like_CXCL13_helper_T",
"4"  = "Naive_Memory_T",
"5"  = "Naive_Memory_T",
"6"  = "Exhausted_CD8_T",
"7"  = "Activated_memory_T",
"8"  = "Cytotoxic_CD8_T",
"9"  = "Treg",
"10" = "Cytotoxic_CD8_T",
"11" = "IFN_response_T",
"12" = "Tfh_like_CXCL13_helper_T",
"13" = "NK_cell",
"14" = "Exhausted_CD8_T",
"15" = "Naive_Memory_T",
"16" = "Treg",
"17" = "Naive_Memory_T")
DimPlot(object =sce.rename, pt.size=0.5,
        reduction = "umap",label = T,split.by = "type") +
  ggsci::scale_color_d3()
sce.rename@meta.data$celltype1 <- Idents(sce.rename)
Idents(sce) <- sce@meta.data$seurat_clusters


library("ggplot2")
# 计算每一类样本中不同细胞的比例并画图
meta_data <- sce.rename@meta.data 
plot_data <- data.frame(table(meta_data$type,meta_data$celltype1))
plot_data$Total <- apply(plot_data,1,function(x)sum(plot_data[plot_data$Var1 == x[1],3]))
plot_data <- plot_data %>% mutate(Percentage = round(Freq/Total,3) * 100)
ggplot(plot_data,aes(x = Var1,y = Percentage,fill = Var2)) +
  geom_bar(stat = "identity",position = "stack") +
  theme_classic() + 
  theme(axis.title.x = element_blank()) + labs(fill = "Cluster")

#######################################################

library(Seurat)
library(dplyr)
library(ggplot2)
library(ggpubr)

df <- FetchData(
  object = obj,
  vars = c("LGALS9", "type", "patients")
)

df2 <- df %>%
  filter(type %in% c("CA", "nor")) %>%
  filter(!is.na(LGALS9), !is.na(type), !is.na(patients))

patient_df <- df2 %>%
  group_by(patients, type) %>%
  summarise(
    LGALS9_mean = mean(LGALS9, na.rm = TRUE),
    LGALS9_median = median(LGALS9, na.rm = TRUE),
    n_cells = n(),
    .groups = "drop"
  )

patient_df$type <- factor(patient_df$type, levels = c("nor", "CA"))

ggplot(patient_df, aes(x = type, y = LGALS9_mean, fill = type)) +
  geom_boxplot(width = 0.6, outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.15, size = 2.5, alpha = 0.9) +
  stat_compare_means(
    method = "wilcox.test",
    label = "p.format"
  ) +
  labs(
    x = NULL,
    y = "Mean LGALS9 expression per patient",
    title = "LGALS9 expression in nor vs CA (patient-level)"
  ) +
  theme_classic(base_size = 14) +
  theme(legend.position = "none")

###################################################
df$celltype1 <- factor(df$celltype1, levels = c("nor", setdiff(unique(as.character(df$celltype1)), "nor")))
library(Seurat)
library(ggplot2)
library(ggpubr)

DefaultAssay(obj) <- "RNA"

df <- FetchData(obj, vars = c("LGALS9", "celltype1"))
df <- na.omit(df)

df$celltype1 <- factor(
  df$celltype1,
  levels = c("nor", setdiff(unique(as.character(df$celltype1)), "nor"))
)

ggplot(df, aes(x = celltype1, y = LGALS9, fill = celltype1)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.8) + 
  stat_compare_means(
    method = "kruskal.test",
    label = "p.format"
  ) +
  theme_classic(base_size = 14) +
  labs(
    x = "celltype1",
    y = "LGALS9 expression",
    title = "LGALS9 expression across celltype1 groups"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )
