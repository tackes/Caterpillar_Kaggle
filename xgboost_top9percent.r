########################
# Author: Jeff Tackes  #
# Caterpillar Kaggle   #
# March 2018           #
# Ranking Top 9%       #
########################



#install.packages('Ckmeans.1d.dp')
#install.packages("https://cran.r-project.org/bin/windows/contrib/3.5/rlang_0.2.0.zip")

set.seed(2)

library(data.table)
library(Ckmeans.1d.dp)

### Change base to the folder name of your data
base = "C:/Users/jeffrey.tackes/Desktop/Predict 454 Data/Final/Data/"
test = read.csv(paste0(base, "test_set.csv"))
train = read.csv(paste0(base, "train_set.csv"))

train$id = -(1:nrow(train))
test$cost = 0

dFull = rbind(train, test)

dFull = merge(dFull, read.csv(paste0(base, "bill_of_materials.csv"), quote = ""), by = "tube_assembly_id", all.x = TRUE)
dFull = merge(dFull, read.csv(paste0(base, "specs.csv"), quote = ""), by = "tube_assembly_id", all.x = TRUE)
dFull = merge(dFull, read.csv(paste0(base, "tube.csv"), quote = ""), by = "tube_assembly_id", all.x = TRUE)
compFiles = dir(base)[grep("comp_", dir(base))]

idComp = 1
keyMerge = 0
for(idComp in 1:8){
  for(f in compFiles){
    d = read.csv(paste0(base, f), sep = ',', quote = "")
    names(d) = paste0(names(d), "_", keyMerge)
    dFull = merge(dFull, d, by.x = paste0("component_id_", idComp), by.y = paste0("component_id_", keyMerge), all.x = TRUE)
    keyMerge = keyMerge + 1
  }
  cat("idComp = ", idComp, " - nrow(dFull) = ", nrow(dFull), " and ncol(dFull) = ", ncol(dFull), "\n")
}


#######################
# FEATURE ENGINEERING #
#######################

# Adding new features date, year and month
final.data <- as.data.table(dFull)
final.data$quote_date=as.Date(final.data$quote_date)
final.data$quote_year = as.numeric(format(final.data$quote_date, "%Y"))
final.data$quote_month = as.numeric(format(final.data$quote_date, "%m"))
final.data$quote_day_of_month = as.numeric(format(final.data$quote_date, "%d"))
final.data$quote_day_of_week = as.numeric(format(final.data$quote_date, "%w"))+1 # Sunday is 0: values 0-6 - add 1 to each
final.data$quote_week_of_year = as.numeric(format(final.data$quote_date, "%U"))+1 # Sunday is First Day: values 0-52 - add 1 to each




## Mat volume
final.data$vol_full<-final.data$diameter^2*3.1415957/4*final.data$length
final.data$vol_wall<-(final.data$diameter^2-(final.data$diameter-final.data$wall)^2)*3.1415957/4*final.data$length


#####################################################
# Component aggregation & reduction of common data  #
#####################################################

# reduce dataset to only the component properties
# Get all the numeric columns so we can create a sum variable
comp_numeric <- final.data[,names(final.data)[which(sapply(final.data, is.numeric))], with=FALSE] 
final.data_comp <- comp_numeric[ , grepl("_[0-9]" , names( comp_numeric ) ) , with=FALSE]

temp_vect <- c()
for (i in names(final.data_comp)){
  char_num <- attr(regexpr("_[^_]*$", i), "match.length")
  clean_name <- substr(i, 1,nchar(i)-char_num)
  temp_vect[i]<- clean_name
}
remove_cols <- colnames(final.data_comp)
colnames(final.data_comp)<- temp_vect

# Get their column names
noms1 = remove_cols
noms <- names(final.data_comp)
# Remove the last 2 characters so all like variables have exactly the same name
#noms = substr(noms1,1,nchar(noms1)-2) # remove Comp_ID separator so we can add on like names
#Apply those new column names

unique_cols <- unique(noms)

# DEFINE MY ROWWISE FUNCTIONS to calculate summary statistics for each comp. spec
my.max <- function(x) ifelse( !all(is.na(x)), max(x, na.rm=T), NA)
my.min <- function(x) ifelse( !all(is.na(x)), min(x, na.rm=T), NA)
my.sum <- function(x) ifelse( !all(is.na(x)), sum(x, na.rm=T), NA)
my.mean <- function(x) ifelse( !all(is.na(x)), mean(x, na.rm=T), NA)

final.data_comp <- data.frame(final.data_comp)
df_total = data.frame(1:60448,1)

# Loop through all like name variables and calculate MAX, MIN, MEAN, SUM and add to a DF
for (colname in unique_cols) {
  temp_df = data.frame()
  temp1 <- comp_numeric[,grep(colnames(comp_numeric),pattern=colname,fixed = TRUE), with=FALSE]
  max_name <- paste(colname,"_max", sep="")
  min_name <- paste(colname,"_min", sep="")
  mean_name <- paste(colname,"_mean", sep="")
  sum_name <- paste(colname,"_sum", sep="")
  
  max <- as.data.frame(apply(temp1, 1, my.max))
  names(max)[names(max) =="apply(temp1, 1, my.max)"]<-max_name 
  min <- as.data.frame(apply(temp1, 1, my.min))
  names(min)[names(min) =="apply(temp1, 1, my.min)"]<-min_name
  mean <- as.data.frame(apply(temp1, 1, my.mean))
  names(mean)[names(mean) =="apply(temp1, 1, my.mean)"]<-mean_name
  sum <- as.data.frame(apply(temp1, 1, my.sum))
  names(sum)[names(sum) =="apply(temp1, 1, my.sum)"]<-sum_name
  temp_df <- cbind(max,min,mean,sum)
  df_total <- cbind(df_total, temp_df)
  
}

# Convert TUBE ASSEMBLY ID to NUMERIC
#final.data$tube_assembly_id<- as.integer(gsub('TA-',"",final.data$tube_assembly_id))


drop <- c('X1.60448','X1') 
df_total_1 <- df_total[,!(names(df_total) %in% drop)]

final.data <- final.data[,!(names(final.data) %in% noms1), with=FALSE]
final.data <- cbind(final.data, df_total_1)


backup<- final.data
final.data <- backup


## Tube assemblies
final.data$tube_assembly_id<-as.numeric(substr(final.data$tube_assembly_id,4,8))

final.data <- as.data.frame(final.data)
### Clean NA values
for(i in 1:ncol(final.data)){
  if(is.numeric(final.data[,i])){
    final.data[is.na(final.data[,i]),i] = -1
  }else{
    final.data[,i] = as.character(final.data[,i])
    final.data[is.na(final.data[,i]),i] = "NAvalue"
    final.data[,i] = as.factor(final.data[,i])
  }
}


### Clean variables with too many categories
for(i in 1:ncol(final.data)){
  if(!is.numeric(final.data[,i])){
    freq = data.frame(table(final.data[,i]))
    freq = freq[order(freq$Freq, decreasing = TRUE),]
    final.data[,i] = as.character(match(final.data[,i], freq$Var1[1:30]))
    final.data[is.na(final.data[,i]),i] = "rareValue"
    final.data[,i] = as.factor(final.data[,i])
  }
}

final.data$log_cost <- log(final.data$cost+1)

test = final.data[which(final.data$id > 0),]
train = final.data[which(final.data$id < 0),]

set.seed(123)
#smp_size<-floor(0.8*nrow(train)) #80% will be training and 20% will be validation
#train_ind<-sample(seq_len(nrow(train)),size=smp_size)
#data.train<-train[train_ind,] #this is the dataset to train the model on
#data.validation<-train[-train_ind,] #this is the dataset to check model accuracy on 

#### 
#EDA
train_eda <- final.data[which(final.data$id < 0),]
library(ggplot2)
g <- ggplot(train_eda,aes(x=quantity, y=cost)) + #,color = supplier)) + 
  geom_point(alpha=0.6)+ 
  scale_x_log10() + scale_y_log10() + 
  stat_smooth(method=lm)+
  labs(title="Avg Tube Price per Volume Order")
print(g)

y <- ggplot(train_eda[train_eda$tube_assembly_id == 'TA-02308'],aes(x=quote_year, y=cost,color = supplier, size=quantity)) + #,color = supplier)) + 
  geom_point(alpha=0.6)+ 
  
  stat_smooth(method=lm)+
  labs(title="Avg Price TA-02308")
print(y)


data.train <- train
data.test <- test



######################
# RANDOM FOREST MODEL#
######################
library(h2o)
h2o.init(max_mem_size='1000G')
train_h2o <- as.h2o(data.train)
test_h2o <- as.h2o(data.test)
library(randomForest)
set.seed(1)
features <- colnames(data.train)[!(colnames(data.train) %in% c("id","cost"))]
rf = h2o.randomForest(y="log_cost", x=features, ntree = 100, training_frame = train_h2o)
#rf_backup <- rf
pred_rf = exp(predict(rf, test_h2o)) - 1


###########
# XGBOOST #
###########

library(xgboost)
require(Matrix)
library(purrr)
xg_df <- as.data.frame(final.data)
output_vector = as.numeric(xg_df$log_cost)

xg_test = xg_df[which(xg_df$id > 0),]
xg_train = xg_df[which(xg_df$id < 0),]

xg_train <- as.data.frame(xg_train)
xg_test<- as.data.frame(xg_test)

xg_train1<- xg_train[ , !(names(xg_train) %in% c('cost','log_cost'))]
xg_test1<- xg_test[ , !(names(xg_test) %in% c('cost','log_cost'))]
xg_test1[] <- lapply(xg_test1, as.numeric)
xg_train1[] <- lapply(xg_train1, as.numeric)

xg_train1 <- as.matrix(xg_train1)
xg_test1 <- as.matrix(xg_test1)

sparse_matrix <- xgb.DMatrix(data = xg_train1, label = xg_train$log_cost)
sparse_matrix_test <- xgb.DMatrix(data = xg_test1)

bst1 <- xgboost(data = sparse_matrix, label = output_vector, max.depth = 6,
               eta = .05, nthread = 8, nround = 1000,objective = "reg:linear",
               missing = -1, min_child_weight = 5,
               subsample = .6,
               colsample_bytree = .6,
               #scale_pos_weight = sumwneg / sumwpos,
               booster = "gbtree"
                
               )

bst2 <- xgb.cv(data = sparse_matrix, label = output_vector, max.depth = 6,
                eta = .05, nthread = 8, nround = 1000,objective = "reg:linear",
                missing = -1, min_child_weight = 5,
                subsample = .6,
                colsample_bytree = .6,
                #scale_pos_weight = sumwneg / sumwpos,
                booster = "gbtree",
                metrics = list("rmse"),
                nfold=3
                
)

library(DiagrammeR)
library(viridisLite)
library(viridis)

# simple tree
bst5 <- xgboost(data = sparse_matrix, label = output_vector, max.depth = 2,
                eta = 1, nthread = 1, nround = 100,objective = "reg:linear",
                missing = -1, min_child_weight = 3,
                subsample = .6,
                colsample_bytree = .6,
                #scale_pos_weight = sumwneg / sumwpos,
                booster = "gbtree"
                
)

pdf(file = "XGTree2.pdf"  )


node_attrs <- c("shape = circle",
                "fixedsize = TRUE",
                "width = 1",
                "penwidth = 1",
                "color = DodgerBlue",
                "style = filled",
                "fillcolor = Aqua",
                "alpha_fillcolor = 0.5",
                "fontname = Helvetica",
                "fontcolor = Black")


xgb.plot.tree(feature_names = colnames(sparse_matrix),model=bst5, 
                     trees=3, plot_width = 2000, plot_height = 1000,
                     features_keep = 3,n_first_tree=3)




xgb.dump(bst5, 'xgb.model.dump', fname = 'xgboost_dump', fmap = "", with_stats = TRUE, trees=3,dump_format = c("text"),n_first_tree=3)


.

tree_values <- xgb.dump(model=bst5, trees=3)

export_csv(tree)
export_graph(tree, XGB_tree,file_type='pdf')
pdf()

######################
# FEATURE IMPORTANCE #
######################
par(mar=c(1,1,1,1))
importance_matrix <- xgb.importance(colnames(sparse_matrix), model = bst1)

feature_imp <- xgb.plot.importance(importance_matrix, top_n = 20,
                          xlab = "Relative importance")
write.csv(feature_imp, "XGBoost_featureImportance.csv", row.names = FALSE, quote = FALSE)

(gg <- xgb.ggplot.importance(importance_matrix = importance, measure = 'Gain', rel_to_first = TRUE,top_n = 10))
gg + ggplot2::ylab("Gain")
(gg <- xgb.ggplot.importance(importance_matrix = importance, measure = 'Frequency', rel_to_first = TRUE,top_n = 10))
gg + ggplot2::ylab("Frequency")
(gg <- xgb.ggplot.importance(importance_matrix = importance, measure = 'Importance', rel_to_first = TRUE,top_n = 10))
gg + ggplot2::ylab("Importance")
(gg <- xgb.ggplot.importance(importance_matrix = importance, measure = 'Cover', rel_to_first = TRUE,top_n = 10))
gg + ggplot2::ylab("Cover")


# eta - learning paratmeter step
# nthread - number of computing cores
# max.depth - maximum depth before spliting of nodes

pred_xg <- exp(predict(bst1, sparse_matrix_test))-1


############
# ENSEMBLE #
############
library(ModelMetrics)
pred_rf <- as.vector(pred_rf)
rmse(data.validation$cost, (pred_rf + pred_xg)/2)

pred_rf <- as.vector(pred_rf)
pred_ensemble <- (pred_rf + pred_xg)/2


# OUTPUT FILE
pred_df <- as.data.frame(pred_xg)
submitDb = data.frame(id = data.test$id, cost = pred_df)
colnames(submitDb)<- c('id','cost')
write.csv(submitDb, "submit_xg.001_25000.csv", row.names = FALSE, quote = FALSE)


