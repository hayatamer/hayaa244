clear all
data=xlsread('C:\Users\Tamer Montaser\Documents\MATLAB\heart_DD.csv');
size(data);
alpha=0.01;
m=length(data);
target=data(:,14);
basic_features=data(:,1:5);
extra_features=data(:,8:10);

%set_features=[ones(m,1) basic_features extra_features basic_features.^2 basic_features.^3];
set_features=[ones(m,1) basic_features];
%set_features=[ones(m,1) basic_features basic_features.^2 basic_features.^3];
%set_features=[ones(m,1) basic_features extra_features basic_features.^2];
n=length(set_features(1,:));

for w=2:n
    if max(abs(set_features(:,w)))~=0
    set_features(:,w)=(set_features(:,w)-mean((set_features(:,w))))./std(set_features(:,w)); %average of the X, scaling 
    end
end

scale_target=data(:,14)/mean(data(:,14)); %scaling the y
theta=zeros(n,1);
k=1;

cost_function(k)=(1/m)*sum((((-1)*scale_target)*(log(set_features*theta))')-((1-scale_target)*(log(1-set_features*theta))'));

%+((lamda/2*m)*(sum(theta).^2))

%Gradient Descent
R=1;
while R==1
theta=theta-(alpha/m)*set_features'*(set_features*theta-scale_target);
k=k+1;
cost_function(k)=(1/(m))*(sum(-scale_target*(log(set_features*theta))')-(1-scale_target)*(log(1-(set_features*theta)))');
if cost_function(k-1)-cost_function(k)<0
    break
end 
error=(cost_function(k-1)-cost_function(k))./cost_function(k-1);
if error <.0001;
    R=0;
end
end

plot(cost_function)