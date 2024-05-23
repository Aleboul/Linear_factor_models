source('muscle.R')
library(wordspace)
library(skmeans)

X = read.csv(paste('results_model_1/data/data_',1,'.csv', sep=''))

X = X[,-1]
d = ncol(X)
n = nrow(X)
k = as.integer(sqrt(n))

norms = rowNorms(as.matrix(X), method = "euclidean", p = 2)
indices = rank(norms) > n-k
norms_ext = as.matrix(X[indices,])
km <- skmeans(norms_ext, 6)
centers = data.frame(km$prototypes)
write.csv(centers,file=paste("results_model_1/results_skmeans/centers_",1,".csv", sep=''), row.names=F)
