%% Matlab implementation of the Unsupervised K-Means Clustering Algorithm
% Written by Kristina P. Sinaga (kristinasinaga57@yahoo.co.id)

%% INPUTS:
% points : input data matrix. 
% cluster_n: number of clusters


%% Notation:
% points ... a cell containing all views for the data
% label ... ground truth labels
% clust ... predict labels
% u .... the data point i belongs to k-th cluster
% alpha ... the probability of one data point belonged to the kth class

%% Parameters

% gamma ... exp(-cluster_n/250); exp(-cluster_n/300); exp(-cluster_n/500)
% eta ... min(1,a^floor(points_dim/2-1))

clear all;close all;clc


load fisheriris;
points=meas;
y=categorical(species);
label=grp2idx(y);


%% DEFINING INPUT DATA 

points_n   = size(points,1);
points_dim = size(points,2);

thres = 0.1;
beta  = 1;
gamma = 1;
rate  = 0;

%% Initializations 

cluster_n = points_n;
clust_cen = points+0.0001; 
alpha     = ones(1,cluster_n)*1/cluster_n; 
err       = 10;
 
t_max     = 100;


c_history=[];

while and(cluster_n>1,err>=thres)
% for itr=1:8
    
    rate = rate + 1;
    
    %% Step 2 : Compute membership
    
    u=zeros(points_n,cluster_n);
    D7=[];
    for k=1:cluster_n
        D1=bsxfun(@minus,points,clust_cen(k,:));
        D2=D1.^2;
        D3=sum(D2,2);
        D4=D3;
        D5=gamma*log(alpha(k));
        D6=bsxfun(@minus,D4,D5);
        D7=[D7 D6];
    end
    
    if rate==1
       D8=D7;
       D7(logical(eye(size(D7))))=NaN;
       [val idx]=min(D7,[],2);
       D7(isnan(D7))=diag(D8);
    else
       [val idx]=min(D7,[],2);
    end
    
    for i=1:points_n
        u(i,idx(i))=1;
    end
    u;
    
    
    %% STEP 3: Compute gamma
    
    gamma = exp(-cluster_n/450);
    
    %% STEP 4: Update alpha 
    
    new_alpha=sum(u,1)/points_n+beta/gamma*alpha.*(log(alpha)-sum(alpha.*log(alpha)));
    
    %% STEP 5: Compute beta
    
    a=1/rate;
    eta=min(1,a^floor(points_dim/2-1));
    
    temp9=0; 
    for k=1:cluster_n
        temp8=exp(-eta*points_n*abs(new_alpha(k)-alpha(k)));
        temp9=temp9+temp8;
    end
    temp9=temp9/cluster_n;
    temp10=1-max(sum(u,1)/points_n);
    temp11=sum(alpha.*log(alpha));
    temp12=temp10/(-max(alpha)*temp11);

    new_beta=min(temp9,temp12);
    
    
    %% STEP 6: Update number of clusters
    
    index=find(new_alpha<=1/points_n);
    
    %% ADJUST ALPHA
    adj_alpha=new_alpha;
    adj_alpha(index)=[];
    adj_alpha=adj_alpha/sum(adj_alpha);
    new_alpha=adj_alpha;
    if size(new_alpha,2)==1
        new_alpha=alpha;
        break;
    end
    
    %% UPDATE NUMBER OF CLUSTER
    new_cluster_n=size(new_alpha,2);
    
    %% ADJUST MEMBERSHIP (U)
    adj_u=u;
    adj_u(:,index)=[];
    adj_u=bsxfun(@rdivide,adj_u,sum(adj_u,2));
    adj_u(isnan(adj_u))=0;
    new_u=adj_u;
    
    if and(rate>=60,new_cluster_n-cluster_n==0)
        new_beta=0;
    end
    
    
    %% Update Cluster Centers
    
    new_clust_cen=[];
    for k=1:new_cluster_n
            temp4=zeros(1,points_dim);
            temp5=0;
            for i=1:points_n
                temp4=temp4+new_u(i,k)*points(i,:);
                temp5=temp5+new_u(i,k);
            end
        new_clust_cen=[new_clust_cen; temp4/temp5];
    end
    new_clust_cen(isnan(new_clust_cen))=sum(mean(points));
    
    
    %% STEP 8: Convergence criteria

    error=[];
    for k=1:new_cluster_n    
        error=[error;norm(new_clust_cen(k,:)-clust_cen(k,:))];
    end   
    
    err=max(error);
    
    
    clust_cen=new_clust_cen;
    cluster_n=new_cluster_n;
    alpha=new_alpha;
    beta=new_beta;
    u=new_u;
    c_history=[c_history cluster_n];
    
end



cluster_n


clust=[];
for i=1:points_n
    [num idx]=max(u(i,:));
    clust=[clust;idx];
end

AR   =1-ErrorRate(label,clust,cluster_n)/points_n
