n = 256; 
l = 4; 
A = zeros(n); 

for i=1:n 
    for j=1:n 
        A(i,j) = spiral((i/n-0.5)*l,(j/n-0.5)*l); 
        A(i,j) = A(i,j) * exp(-((i-n/2)^2+(j-n/2)^2)/7000);
        A(i,j) = A(i,j) * (1- exp(-((i-n/2)^2+(j-n/2)^2)/500));
    end 
end 

imshow(A)
A = A / sum(A(:));
exrwritechannels(['vortex' '.exr'],'piz','single','Y',A);

function val = spiral(x,y) 

    s= 0.8; 

    r = sqrt(x*x + y*y); 
    a = atan2(y,x)*4+r*r;     

    x = r*cos(a);
    y = r*sin(a); 
    
    val = 0; 
    if (abs(x)<s) 
        val = s - abs(x); 
    end
    
    if (abs(y)<s)
        val = max(s-abs(y),val); 
    end 

%     val = 1/(1+exp(-1*(val))); 

end