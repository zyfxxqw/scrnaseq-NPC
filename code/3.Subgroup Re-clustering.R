
experiment.aggregate <- readRDS("1-33raw.rds")
DefaultAssay(obj)
DefaultAssay(obj) <- "RNA"


cells_sub <- subset(experiment.aggregate@meta.data, 
                    new_celltype %in% c("Fibroblasts"))
scRNA_sub <- subset(experiment.aggregate, 
                    cells=row.names(cells_sub))

experiment.aggregate<- subset(experiment.aggregate,
                              idents = unique(Idents(experiment.aggregate))[!unique(Idents(experiment.aggregate)) %in% c(20,16)]
)

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




