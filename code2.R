rm(list=ls())
gc(reset=T)
library(jpeg)
library(readr) # 빨리 읽을 수 있는 패키지
library(deepnet) # deep neural net 패키지
library(caret)

# edge 찾는 함수, edge를 찾아서 그림을 2진으로 만듦

#### 문제점 ( 그냥 k-means로 분류 )
image = readJPEG("전화번호2.jpg")
image = (image[,,1] + image[,,2] + image[,,3])/3
detect_edge = function(image) ## K-means clustering 을 이용한 테두리 추출
{
  ylen<-dim(image)[1]
  xlen<-dim(image)[2]
  ##대각선 왼쪽 위에서 오른쪽 아래  
  image1 <- image[1:(ylen-1),1:(xlen-1)]  #Red
  image2 <- image[2:ylen,2:xlen]
  ## 대각선 오른쪽 위에서 왼쪽아래
  image3 <- image[1:(ylen-1),2:xlen]    #Red
  image4 <- image[2:ylen,1:(xlen-1)]     
  ## 왼쪽 오른쪽 비교
  image5 <- image[1:(ylen-1),1:(xlen-1)] #Red
  image6<- image[1:(ylen-1),2:xlen]     
  ## 위 아래 비교
  image7 <- image[1:(ylen-1),1:(xlen-1)] #Red
  image8 <- image[2:ylen,1:(xlen-1)]     
  #결과행렬 만들기
  result1 <-sqrt((image1-image2)^2)#거리 차이가 심해지도록 가중치 설정
  result2 <-sqrt((image3-image4)^2)
  result3 <-sqrt((image5-image6)^2)
  result4 <-sqrt((image7-image8)^2)
  result<- (result1 + result2 + result3 + result4)/4 # 모든 변화량의 평균을 계산
  kmeans_result <- kmeans(as.vector(result),2)$cluster # 2개의 묶음으로 군집하여 값만 가져옴
  if(sum(kmeans_result == 1) >= sum(kmeans_result== 2)) # 배경과 선을 갯수로 판단하여 선에는 1을 주고 배경에는 0을 줌
  {
    kmeans_result <- gsub("1","0",kmeans_result)
    kmeans_result <- gsub("2","1",kmeans_result)
  }else
  {
    kmeans_result <- gsub("2","0",kmeans_result)
    kmeans_result <- gsub("1","1",kmeans_result)
  }
  
  finalresult <- matrix(kmeans_result,dim(result))
  return(finalresult)
}
rotate = function(image) ## rotation using PCA
{
  row_index = which(image==1) %% nrow(image) # 행
  col_index = which(image==1) %/% nrow(image)+1 # 열
  image_matrix = cbind(row_index,col_index)
  head(image_matrix)
  pca_image = prcomp(image_matrix,center=TRUE) # PCA를 이용한 회전
  rotated_image = pca_image$x[,1:2] # 회전후의 데이터
  ## 원래 상태로 되돌리기
  min_vec = c(min(rotated_image[,1]),min(rotated_image[,2]))
  revert_image = rotated_image - matrix(min_vec,nrow=nrow(rotated_image),ncol=ncol(rotated_image),byrow=T)
  revert_image = round(revert_image) # 좌표를 반올림하여 찍어줌
  max_vec = c(max(revert_image[,1]),max(revert_image[,2]))
  origin_image = matrix(0,nrow=max_vec[2],ncol=max_vec[1])
  for(i in 1:nrow(revert_image))
  {
    origin_image[max_vec[2]-revert_image[i,2],revert_image[i,1]] = 1
  }
  return(list(rotated=origin_image,rotation=pca_image$rotation))
}
hog = function(image,window=3) ## Histogram of Oriented Gradient, gray image만 input
{
  image = image[1:(nrow(image)-nrow(image)%%window),1:(ncol(image)-ncol(image)%%window)] # image를 window로 자르기 쉽게 끝 픽셀을 편집함
  row_break = c(0,nrow(image)%/%window * 1:window)
  col_break = c(0,ncol(image)%/%window * 1:window)
  hog_vec = numeric()
  for(i in 1:(window))
  {
    for(j in 1:(window))
    {
      temp_image = as.matrix(image[(row_break[i]+1):row_break[i+1],(col_break[j]+1):col_break[j+1]])
      gradient_x = temp_image[2:(nrow(temp_image)-1),3:ncol(temp_image)] - temp_image[2:(nrow(temp_image)-1),1:(ncol(temp_image)-2)] + 0.0001 # arctan에서 NaN 발생 피하기 위함
      gradient_y = temp_image[3:nrow(temp_image),2:(ncol(temp_image)-1)] - temp_image[1:(nrow(temp_image)-2),2:(ncol(temp_image)-1)] + 0.0001
      gradient = sqrt(gradient_x ^2 + gradient_y ^2)
      arctan = atan(gradient_y/gradient_x)
      arctan_cut = cut(arctan,breaks=seq(-pi/2,pi/2,length=10))
      hist_vec = tapply(gradient,arctan_cut,sum)
      hist_vec[is.na(hist_vec)] = 0
      hist_vec = hist_vec/sqrt(hist_vec^2+0.01^2)
      hog_vec = c(hog_vec,hist_vec)
    }
  }
  return(as.numeric(hog_vec))
}
rasterImage(image,0,0,1,1)
bad =kmeans(c(image),2)
bad_image = matrix(bad$cluster,nrow=nrow(image),ncol=ncol(image))
plot(0:1,0:1,type="n")
rasterImage(bad_image-1,0,0,1,1)
# 왜 edge detection을 해야 하는가?
temp = detect_edge(image)
temp = apply(temp,2,as.numeric)
plot(0:1,0:1,type="n")
rasterImage(temp,0,0,1,1)
###

# read Image, image 파일을 읽는 과정
setwd("C:\\Users\\msj55\\Desktop\\프로젝트\\프로젝트")
getwd()
setwd("./프로젝트")
list.files()
phone = readJPEG("30degree.jpg") 
# phone = readJPEG("120degree.jpg") 
gray = (phone[,,1]+phone[,,2]+phone[,,3])/3 # 흑백그림으로 바꿈
plot(0:1,0:1,type="n")
rasterImage(gray,0,0,1,1) # 입력된 image plot으로 확인

## 다른 edge detection
gray2 = 1-gray
plot(0:1,0:1,type="n")
gray3 = ifelse(gray2<0.403, 0,1)
rasterImage(gray3,0,0,1,1)

image = rotate(gray3)$rotated
plot(0:1,0:1,type="n")
rasterImage(image,0,0,1,1) # 회전 완료된 모습

# edge detection과 plotting
gray_edge = detect_edge(gray)
gray_edge = apply(gray_edge,2,as.numeric)
plot(0:1,0:1,type="n")
rasterImage(gray_edge,0,0,1,1)

image = rotate(gray_edge)$rotated ## 회전된 이미지를 image에 저장
plot(0:1,0:1,type="n")
rasterImage(image,0,0,1,1) # 회전 완료된 모습

# hog를 이용한 digit_ recognizer)
data = read_csv("train.csv") 
plot(0:1,0:1,type="n")
rasterImage(matrix(as.numeric(data[1,-785])/256,nrow=28),0,0,1,1)
nrow(data) # train data 42000개
table(data$label)
# kk = read_csv("test.csv") ## test data 28000개 
# nrow(kk)

index = unlist(createDataPartition(data$label,p=0.7))
train = data[index,]
test = data[-index,]
label = train$label
test_label = test$label
train = train[,-1]/256
test = test[,-1]/256
window = 3
final_train = matrix(0,ncol=window^2 * 9,nrow=nrow(train))
final_test = matrix(0,ncol=window^2 * 9 ,nrow=nrow(test))
for(i in 1:nrow(train))
{
  final_train[i,] = hog(matrix(as.numeric(train[i,]),nrow=28,ncol=28),window=window)
}
for(i in 1:nrow(test))
{
  final_test[i,] = hog(matrix(as.numeric(test[i,]),nrow=28,ncol=28),window=window)
}
hog_train = cbind(final_train,label)
hog_test = cbind(final_test,test_label)
colnames(hog_train)[1:81] = paste0("hog",1:81)
colnames(hog_test)[1:81] = paste0("hog",1:81)
write.csv(hog_train,"hog_train3.csv",row.names=F)
write.csv(hog_test,"hog_test3.csv",row.names=F)

### 여기서부터 숫자 인식 과정
hog_train = read_csv("hog_train3.csv")
hog_test = read_csv("hog_test3.csv")

## nnet
library(e1071)
fit_svm = svm(factor(label)~.,data=hog_train)
pred = predict(fit_svm,hog_test,type="class")

library(nnet)
fit = nnet(factor(label)~.,data=hog_train,size=10)
pred = predict(fit,hog_test,type="class")
true = hog_test$test_label
sum((pred==true)/length(true)) ## 87 % 10개의 노드로 적합한 신경망

## bagging + nnet
# 모형 만들기
for(i in 1:10)
{
  sample = unlist(createDataPartition(hog_train$label,p=0.7)) # boostrap sampling
  temp = hog_train[sample,]
  eval(parse(text=paste0("fit",i,"= svm(factor(label)~.,data=temp)")))
  cat(i,"\n")
}
# 예측 하기
final_model = function(data) ## hog 형태의 input으로 판별
{
  pred = list()
  for(i in 1:17)
  {
    eval(parse(text=paste0("pred[[",i,"]]= predict(fit_svm",i,",data,type=\"class\")")))
  }
  pred_matrix = data.frame(pred)
  pred_matrix = as.matrix(pred_matrix)
  colnames(pred_matrix) = NULL
  which.many = function(x)
  {
    x = as.numeric(x)
    result = names(which.max(table(x)))
    if(length(result)==2) result = result[1]
    return(names(which.max(table(x))))
  }
  pred_final = as.numeric(apply(pred_matrix,1,which.many))
  return(pred_final)
}

pred = final_model(hog_test)
sum(true==pred)/length(true) # 89 % 

### 해보자 한번
image = rotate(gray_edge)$rotated ## 회전된 이미지를 image에 저장
image = rotate(gray3)$rotated
image_col = colSums(image)
plot(image_col,type="l")
ind = which(abs(diff(image_col)) > 3)
abline(v=ind,col="red")
k_ind = kmeans(ind,3)$cluster
break1 = tapply(ind,k_ind,min)
break2 = tapply(ind,k_ind,max)

plot(image_col,type="l")
abline(v=c(break1,break2),col="red",lwd=2)
breaks = cbind(break1,break2)
part1 = image[,breaks[1,1]:breaks[1,2]]
part2 = image[,breaks[2,1]:breaks[2,2]]
part3 = image[,breaks[3,1]:breaks[3,2]]

breaks
# 잘린 부분 표시
plot(0:1,0:1,type="n")
rasterImage(image,0,0,1,1)
breaks = breaks[order(breaks[,1]),]
rect(breaks[1,1]/ncol(image),0,breaks[1,2]/ncol(image),1,border="red",lwd=5)
rect(breaks[2,1]/ncol(image),0,breaks[2,2]/ncol(image),1,border="red",lwd=5)
rect(breaks[3,1]/ncol(image),0,breaks[3,2]/ncol(image),1,border="red",lwd=5)
rasterImage(part1,0,0,1,1)
rasterImage(part2,0,0,1,1)
rasterImage(part3,0,0,1,1)
breaks
plot(0:1,0:1,type="n")
rasterImage(image[,3+0:77],0,0,1,1)
hog_image = hog(image[,3:80])
rasterImage(image[,81+0:77],0,0,1,1)
hog_image = hog(image[,81+0:77])
rasterImage(image[,:80],0,0,1,1)
hog_image = hog(image[,3:80])
rasterImage(image[,3:80],0,0,1,1)


plot(0:1,0:1,type="n")
rasterImage(image[,410+0:81],0,0,1,1)
hog_image = hog(image[,3:80])
hog_image = data.frame(matrix(hog_image,nrow=1))
colnames(hog_image) = paste0("hog",1:81)
predict(fit_svm,hog_image,type="class")



str(matrix(data[1,-785]/256,nrow=28,ncol=28))
rasterImage(matrix(as.numeric(data[1,-785]/256),nrow=28,ncol=28),0,0,1,1)

## boosting
fit = boosting(label ~ ., data = hog_train, boos = TRUE, mfinal = 10, control = (minsplit = 0))
pred = predict(fit,hog_test)
pred = predict(fit,hog_train)
sum(as.numeric(pred$class)==true)/length(true)
sum(as.numeric(pred$class)==hog_train$label)/length(hog_train$label) ## 87 %


## hog deep learning training
train_mat = data.matrix(hog_train)
test_mat = data.matrix(hog_test)
y_train = as.factor(hog_train$label)
y_test = as.factor(hog_test$test_label)
train_mat = train_mat[,-145]
test_mat = test_mat[,-145]
outnode_train <- model.matrix( ~ y_train -1)
outnode_test <- model.matrix( ~ y_test -1)
t.start <- Sys.time()
nn <- dbn.dnn.train(train_mat,outnode_train, 
                    hidden=c(500,500,250,125),
                    output="softmax",
                    batchsize=100, numepochs=100, learningrate = 0.1)
t.end <- Sys.time() # 24시간 6분
save(nn, file="hogdigit.RData")
train_pred <- nn.predict(nn, hog_train[,-82])
train_pred_num <- apply(train_pred, 1, function(x) which(max(x)==x))-1

table(true=y_train,pred=train_pred_num)
sum(y_train==train_pred_num)/length(train_pred_num)

test_pred <- nn.predict(nn, hog_test[,-82])
test_pred_num <- apply(test_pred, 1, function(x) which(max(x)==x))-1
table(true=y_test,pred=test_pred_num)

sum(y_test==test_pred_num)/length(test_pred_num)