clear all
data=xlsread('C:\Users\Tamer Montaser\Documents\MATLAB\ML\house_prices_data_training_data.csv');
size(data);
alpha=0.01;
lamda=0.01;
m=length(data);
training=ceil(0.6*m);
cv=ceil(0.2*m);
price=data(:,3);
basic_features=data(1:training,4:19);
extra_features=data(1:training,20:21);

set_features=[ones(training,1) basic_features extra_features basic_features.^2 basic_features.^3];
%set_features=[ones(m,1) basic_features];
%set_features=[ones(m,1) basic_features basic_features.^2 basic_features.^3];
%set_features=[ones(m,1) basic_features extra_features basic_features.^2];
n=length(set_features(1,:));

for w=2:n
    if max(abs(set_features(:,w)))~=0
    set_features(:,w)=(set_features(:,w)-mean((set_features(:,w))))./std(set_features(:,w)); %average of the X, scaling 
    end
end

scale_price=data(1:training,3)/mean(data(1:training,3)); %scaling the y
theta=zeros(n,1);
k=1;

cost_function(k)=(1/(2*training))*sum((set_features*theta-scale_price).^2);%+((lamda/2*training)*(sum(theta).^2));

%Gradient Descent
R=1;
while R==1
theta=theta-(alpha/training)*set_features'*(set_features*theta-scale_price);
k=k+1;
cost_function(k)=(1/(2*training))*sum((set_features*theta-scale_price).^2);%+((lamda/2*training)*(sum(theta).^2));
if cost_function(k-1)-cost_function(k)<0
    break
end 
error=(cost_function(k-1)-cost_function(k))./cost_function(k-1);
if error <.0001;
    R=0;
end
end
figure(1)
plot(cost_function)

%Cross Validation
theta_cv=theta;
basic_features_cv=data(training+1:training+cv,4:19);
extra_features_cv=data(training+1:training+cv,20:21);

set_features_cv=[ones(cv,1) basic_features_cv extra_features_cv basic_features_cv.^2 basic_features_cv.^3];
price_cv=data(training+1:training+cv,3)/mean(data(training+1:training+cv,3));

for w=2:length(set_features_cv(1,:))
    if max(abs(set_features_cv(:,w)))~=0
    set_features_cv(:,w)=(set_features_cv(:,w)-mean((set_features_cv(:,w))))./std(set_features_cv(:,w));
    end
end

cost_function_cv(k)=(1/(2*cv))*sum((set_features_cv*theta_cv-price_cv).^2);%+((lamda/2*cv)*(sum(theta_cv).^2));

%Gradient Descent
R=1;
while R==1
theta_cv=theta_cv-(alpha/cv)*set_features_cv'*(set_features_cv*theta_cv-price_cv);
k=k+1;
cost_function_cv(k)=(1/(2*cv))*sum((set_features_cv*theta_cv-price_cv).^2);%+((lamda/2*training)*(sum(theta).^2));
if cost_function_cv(k-1)-cost_function_cv(k)<0
    break
end 
error_cv=(cost_function_cv(k-1)-cost_function_cv(k))./cost_function_cv(k-1);
if error_cv <.0001;
    R=0;
end
end
figure(2)
plot(cost_function_cv)

%Testing
theta_t=theta;
basic_features_t=data(training+cv:end,4:19);
extra_features_t=data(training+cv:end,20:21);

set_features_t=[ones(length(extra_features_t),1) basic_features_t extra_features_t basic_features_t.^2 basic_features_t.^3];
price_t=data(training+cv:end,3)/mean(data(training+cv:end,3));

for w=2:length(set_features_t(1,:))
    if max(abs(set_features_t(:,w)))~=0
    set_features_t(:,w)=(set_features_t(:,w)-mean((set_features_t(:,w))))./std(set_features_t(:,w));
    end
end

cost_function_t(k)=(1/(2*length(extra_features_t)))*sum((set_features_t*theta_t-price_t).^2);%+((lamda/2*extra_features_t))*(sum(theta_t).^2));

%Gradient Descent
R=1;
while R==1
theta_t=theta_t-(alpha/length(extra_features_t))*set_features_t'*(set_features_t*theta_t-price_t);
k=k+1;
cost_function_t(k)=(1/(2*length(extra_features_t)))*sum((set_features_t*theta_t-price_t).^2);%+((lamda/2*training)*(sum(theta).^2));
if cost_function_t(k-1)-cost_function_t(k)<0
    break
end 
error_t=(cost_function_t(k-1)-cost_function_t(k))./cost_function_t(k-1);
if error_t <.0001;
    R=0;
end
end
figure(3)
plot(cost_function_t)

