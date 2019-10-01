res = 128;

mu = [-1 1];
sigma = [1 0.3; 0.3 0.5];

x1 = linspace(-3,3,res);
x2 = linspace(-3,3,res);
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];

y1 = mvnpdf(X,mu,sigma);
y1 = reshape(y1,length(x2),length(x1));

mu = [1 -1];
sigma = [0.2 0.3; 0.3 1];

x1 = linspace(-3,3,res);
x2 = linspace(-3,3,res);
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];

y2 = mvnpdf(X,mu,sigma);
y2 = reshape(y2,length(x2),length(x1));

mu = [1 1];
sigma = [0.5 0.3; 0.3 0.8];

x1 = linspace(-3,3,res);
x2 = linspace(-3,3,res);
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];

y3 = mvnpdf(X,mu,sigma);
y3 = reshape(y3,length(x2),length(x1));


y = (y1+y2+y3);
y = y./sum(y(:));

surf(x1,x2,y)
caxis([min(y(:))-0.5*range(y(:)),max(y(:))])
xlabel('x1')
ylabel('x2')
zlabel('Probability Density')

exrwritechannels('hm1.exr','zip','single','Y',y);
