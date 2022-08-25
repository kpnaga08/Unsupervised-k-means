clear all;close all;clc

load fisheriris;
points=meas;
y=categorical(species);
label=grp2idx(y);

cluster_n=length(unique(label));
points_n=size(points,1);               
points_dim=size(points,2);

rng(25)
rd=randi([1 points_n],1,cluster_n);
clust_cen=points(rd,:);


thres_km=0.00001;
err=10;
rate=0;

while err>=thres_km
    rate=rate+1;
	%compute u
    u=zeros(points_n,cluster_n);
    u3=[];
    for k=1:cluster_n
        u1=bsxfun(@minus,points,clust_cen(k,:));
        u2=u1.^2;
        u3=[u3 sum(u2,2)];
    end
    [val idx]=min(u3,[],2);
    for i=1:points_n
        u(i,idx(i))=1;
    end
    u;
    
    
    %update cluster center
        new_clust_cen=[];
    for k=1:cluster_n
        temp4=zeros(1,points_dim);
        temp5=sum(u,1);
        for i=1:points_n
            temp4=temp4+u(i,k)*points(i,:);
        end
        new_clust_cen=[new_clust_cen;temp4/temp5(k)];
    end
    
    error=[];
    for k=1:cluster_n    
        error=[error;norm(new_clust_cen(k,:)-clust_cen(k,:))];
    end    
    err=max(error);
        
	clust_cen=new_clust_cen;
end

clust=[];
for i=1:points_n
    [num idx]=max(u(i,:));
    clust=[clust;idx];
end


AR=1-ErrorRate(label,clust,cluster_n)/points_n

