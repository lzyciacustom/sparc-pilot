
suppressMessages({library(spacexr); library(Matrix)})
ref <- as(Matrix::readMM("rctd/ref.mtx"), "CsparseMatrix")
rownames(ref) <- read.csv("rctd/ref_genes.csv", header=FALSE)[,1]
cl <- read.csv("rctd/ref_cells.csv"); colnames(ref) <- cl$cell
ct <- factor(cl$type); names(ct) <- cl$cell
nU <- colSums(ref); reference <- Reference(ref, ct, nU)
for (sid in c("TENX29","TENX152","TENX156")) {
  t0 <- Sys.time()
  m <- as(Matrix::readMM(paste0("rctd/",sid,".mtx")), "CsparseMatrix")
  rownames(m) <- read.csv(paste0("rctd/",sid,"_genes.csv"), header=FALSE)[,1]
  co <- read.csv(paste0("rctd/",sid,"_coords.csv")); colnames(m) <- co$barcode
  xy <- data.frame(x=co$x, y=co$y, row.names=co$barcode)
  puck <- SpatialRNA(xy, m, colSums(m))
  r <- create.RCTD(puck, reference, max_cores=24, UMI_min=20)
  r <- run.RCTD(r, doublet_mode="full")
  w <- as.matrix(normalize_weights(r@results$weights))
  write.csv(w, paste0("rctd/",sid,"_rctd_weights.csv"))
  cat(sid, nrow(w), format(Sys.time()-t0), "\n")
}

