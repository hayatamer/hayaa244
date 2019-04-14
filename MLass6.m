clc
clear all
close all

%%%Principle Component Analysis
%%reading data
ds=xlsread('C:\Users\Tamer Montaser\Documents\MATLAB\ML\house_prices_data_training_data.csv');
X=ds(:,4:21);
n=length(X(1,:));

%1)correlation matrix
corr_x=corr(X);

%3)covariance matrix
cov_x=cov(X);

%4)SVD function (principle component analysis)
[U,S,V]=svd(cov_x);

%5)Eigen Value and alpha 
eigen_values=diag(S);
m=length(eigen_values);
for i=1:m
    alpha=1-(sum(eigen_values(1:i))/sum(eigen_values));
    if (alpha<=0.001)
        break
    end
end
K=i;

%6)Reduced data
red_data=(U(:,1:K)')*(X');

%7)Multiplying with the eigen vector
approx_data=U(:,1:K)*red_data;

%8)Error
error=(1/m)*(sum(approx_data-X'));

%9)Linear Regression
h=1;
theta=zeros(m,1);
k=1;
y=X(:,3)/mean(X(:,3));
E(k)=(1/(2*m))*sum((approx_data'*theta-y).^2); %cost function
lamda=0.001;
alpha=0.01;
while h==1
    alpha=alpha*1;
    theta=theta-(alpha/m)*approx_data*(approx_data'*theta-y);
    k=k+1;
    E(k)=(1/(2*m))*sum((approx_data'*theta-y).^2);
    Reg(k)=(1/(2*m))*sum((approx_data'*theta-y).^2)+(lamda/(2*m))*sum(theta.^2); %regularized cost function
    
    if E(k-1)-E(k)<0;
        break
    end
    q=(E(k-1)-E(k))./E(k-1);
    if q <.000001;
        h=0;
    end
end

%%%%Anamoly Detection
mean_data=mean(X);
standard_deviation=std(X);
pdf_data=[];
for i=1:18
   pdf_data=[pdf_data normcdf(X(50,i),mean_data(i),standard_deviation(i))];
end

if prod(pdf_data)>0.999
    anamoly=1;
else 
    if prod(pdf_data)<0.001
        anamoly=0;
    end
end

%%%KMEANS
costFunction = zeros(1,15); 
[m n]=size(X);
centroid=zeros(m,n);
K=3;
for q = 1:5
    initial_index=randperm(m);
    centroid=X(initial_index(1:q),:);
    oldCentroids=zeros(size(centroid));
    indices=zeros(size(X,1), 1);
    distance=zeros(m,q);
    doNotStop=true;
    iterations=0;
    while(doNotStop)
        for i = 1:m
            for j = 1:q
                distance(i, j) = sum((X(i,:) - centroid(j,:)).^2);
            end
        end
        for i = 1:m
            indices(i) = find(distance(i,:)==min(distance(i,:)));
        end
        for i= 1:q
            clustering = X(find(indices == i), :);
            centroid(i, :) = mean(clustering);
            cost = 0; %costfunction
            for z = 1 : size(clustering,1)
                cost = cost + (1/m)*sum((clustering(z,:) - centroid(i,:)).^2);
            end
            costFunction(1,q) = cost;
        end
         if oldCentroids == centroid
            doNotStop = false;
         end
        oldCentroids = centroid;
        iterations = iterations + 1;
        end
    end
[ o ,K_Optimal] = min(costFunction);
numberOFClusters = 1:15;
plot(numberOFClusters, costFunction);



