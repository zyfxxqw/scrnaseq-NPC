library(patchwork)
library(Seurat)
library(ggplot2)
library(cowplot)
library(Matrix)
library(dplyr)
library(ggsci)

experiment.aggregate <- readRDS("1-34raw.rds")

#10X 数据
NPC1 <- Read10X("./SJC/")

colnames(NPC1) <- paste(colnames(NPC1),"NPC1_nor",sep = "_")



experiment.data <- cbind(NPC1,NPC2,NPC3,NPC4,NPC5,NPC6,NPC7,NPC8,NPC9,NPC10,NPC11,NPC12,NPC13,NPC14,NPC15,NPC16,NPC17,
                         NPC18,NPC19,NPC20,NPC21,NPC22,NPC23,NPC24,NPC25,NPC26,NPC27,NPC28,NPC29,NPC30,NPC31,NPC32,NPC33,NPC34)


experiment.aggregate <- CreateSeuratObject(
  experiment.data,
  project = "multi",
  min.cells = 10,
  min.features = 200)


experiment.aggregate[["percent.mt"]] <- PercentageFeatureSet(experiment.aggregate,
                                                             pattern = "^MT-")

cat("Before filter :",nrow(experiment.aggregate@meta.data),"cells\n")
experiment.aggregate <- subset(experiment.aggregate,
                               subset =
                                 nFeature_RNA > 200 &
                                 nFeature_RNA < 8000&
                                 nCount_RNA > 500 & nCount_RNA <100000&
                                 percent.mt < 15)




experiment.aggregate <- NormalizeData(experiment.aggregate,
                                      normalization.method = "LogNormalize",
                                      scale.factor = 10000)


experiment.aggregate <- FindVariableFeatures(experiment.aggregate,
                                             selection.method = "vst",
                                             nfeatures = 2000)


experiment.aggregate <- ScaleData(
  object = experiment.aggregate, features = VariableFeatures(experiment.aggregate),
  vars.to.regress = c("nCount_RNA","percent.mt"))


experiment.aggregate <- RunPCA(object = experiment.aggregate,
                               features = VariableFeatures(experiment.aggregate),
                               verbose = F,npcs = 50)
ElbowPlot(experiment.aggregate, ndims = 50)
library("stringr")


phe=str_split(rownames(experiment.aggregate@meta.data),'_',simplify = T)
head(phe)
experiment.aggregate@meta.data$patients=phe[,2]
experiment.aggregate@meta.data$type=phe[,3]
library(harmony)
experiment.aggregate <- RunHarmony(experiment.aggregate, group.by.vars = "patients",dims.use = 1:15,theta=3)


dim.use <- 1:15




experiment.aggregate <- FindNeighbors(experiment.aggregate, dims = dim.use,reduction = "harmony")
experiment.aggregate <- FindClusters(experiment.aggregate, resolution = 0.3)
experiment.aggregate <- RunUMAP(experiment.aggregate, dims = dim.use,
                                do.fast = TRUE,,reduction = "harmony",n.neighbors=50, min.dist=0.1)




DimPlot(object = experiment.aggregate, pt.size=0.2,label = T,group.by="celltype")
FeaturePlot(experiment.aggregate,features=c("CD3D","MS4A1","MZB1","KRT19","LYZ","CD79A","COL1A1","CPA3","VWF"),ncol = 3)
FeaturePlot(experiment.aggregate,features=c("SPHK1","LHFPL2","VCAN","POSTN","VAT1","COL10A1","C1S","LAMP2","AEBP1","ITIH2","TMEM132A","PPIC","ITIH1","OAS1","ITGA5","CLNS1","LBP","ENPEP","TTC17","CTHRC1"),ncol = 5)
DotPlot(experiment.aggregate, features =c("MKI67","TOP2A","CENPF","PECAM1","RAMP2","FLT1","CLDN5","MS4A2","KIT","GATA2","DCN","COL1A2","COL1A1","THY1","IGHM","IGHA2","IGHG3","CD79A","LYZ","MARCO","FCGR3A","CD68","KRT19","KRT18","CDH1","EPCAM","NKG7","NCAM1","KLRD1","GNLY","TRAC","CD3G","CD3E","CD3D"))+
  RotatedAxis()


experiment.merged <- RenameIdents(
  object = experiment.aggregate,
  "2" = "T_NK",
  "0" = "T_NK",
  "4" = "Epithelial",
  "10"= "Epithelial",
  "16"="Epithelial",
  "3"="Myeloid",
  "1"="B","9"="B","11"="B","5"="B","7"="B",
  "12"="Fibroblasts","19"="Fibroblasts","17"="Fibroblasts","18"="Fibroblasts","8"="Fibroblasts","15"="Fibroblasts",
  "14"="Mast","13"="Endothelial","6"="Cell_cycle_lymphocytes"
)
pdf(paste0("./",sam.name,"/CellCluster-TSNEPlot_Rename_",max(dim.use),"PC.pdf"),width = 5,height = 4)
DimPlot(object = experiment.merged, pt.size=0.5,
        reduction = "umap",label = T) +
  ggsci::scale_color_d3()
dev.off()
experiment.aggregate@meta.data$celltype <- Idents(experiment.aggregate)
Idents(experiment.aggregate) <- experiment.aggregate@meta.data$celltype
library("ggplot2")


meta_data <- sce@meta.data 
plot_data <- data.frame(table(meta_data$type,meta_data$celltype))
plot_data$Total <- apply(plot_data,1,function(x)sum(plot_data[plot_data$Var1 == x[1],3]))
plot_data <- plot_data %>% mutate(Percentage = round(Freq/Total,3) * 100)
ggplot(plot_data,aes(x = Var1,y = Percentage,fill = Var2)) +
  geom_bar(stat = "identity",position = "stack") +
  theme_classic() + 
  theme(axis.title.x = element_blank()) + labs(fill = "Cluster")+
  scale_fill_manual(values = mycols) +
  scale_color_manual(values = mycols)



