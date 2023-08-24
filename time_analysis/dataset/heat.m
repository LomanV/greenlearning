% Solves the 1d heat equation
% The code is adapted from  RMS Danaraj
% https://www.mathworks.com/matlabcentral/fileexchange/75266-heat-equation-1d-finite-difference-solution

clear;
clc;
clf;

% Noise level of the solutions u
noise_level = 0;

% Input Data
L=1;tf=1;
dom = [0,L];
a=@(t) 0;
b=@(t) 0;
dt=1/1000;
dx=1/100;
alpha=0.025*dt/dx^2;

nx=L/dx;
nt=tf/dt;
t1=0:dt:tf;
fX=0:dx:L;

% Initialize data arrays
Nsample = 200;
U = zeros(nx+1, nx+1, Nsample, 1); % Downsample time to have as many time 
% points as space points
F = zeros(nx+1, Nsample, 1);

A=eye(nx+1);
for i=2:nx
    A(i,i-1:i+1)=[alpha -1-2*alpha alpha];
end

for j=1:Nsample
    
    % We sample the forcing term
    g = rand_sine(dom);

    % We solve the equation for g
    u=zeros(nt+1,nx+1);
    u(:,1)=a(t1');
    u(:,end)=b(t1');
    u(1,:)=g(fX);
    u(1,1) = 0;
    u(1,end) = 0;

    % u(t,x) is deternuned sequentially
    for i=2:nt+1
        B=-u(i-1,:)';
            B(1)=u(i,1);
            B(end)=u(i,end);
    
            u1 =(A\B)'; 
        u(i,2:nx) =u1(2:nx);
    end

    % We save the results
    u = downsample(u, 10);
    U(:,:,j) = u;
    F(:,j)   = g(fX);
end
U = reshape(U, [], Nsample);

% We write the .mat file
 save(sprintf('heat_sine10.mat'),"U","F","fX")

function pwlf = rand_pwl(n, dom)
    % Define the range of the function as dom and sample n+1 points uniformly
    % in the domain
    x = unifrnd(dom(1), dom(2), n+1, 1);
    x = sort(x);
    
    % Generate random coefficients for the function
    sigma = 2;
    mu = 0;
    a = sigma*(randn(1,n)+mu);
    
    % Initialize the piecewise linear function
    pwlf = @(t) a(1).*(t-x(1)).*(t<x(2));
    
    % Calculate the increments
    incr = zeros(1, n-1);
    incr(1) = a(1).*(x(2)-x(1));
    for i = 2:n-1
        incr(i) = incr(i-1) + a(i).*(x(i+1)-x(i));
    end
    
    % Add each piece of the function to the overall function
    for i = 2:n-1
        pwlf = @(t) pwlf(t) + (a(i).*(t - x(i)) + incr(i-1)).*(t >= x(i) & t < x(i+1));
    end
    
    pwlf = @(t) pwlf(t) + (a(n).*(t - x(n)) + incr(n-1)).*(t >= x(n));
end


function f = rand_sine(dom)
    n_el = 10;
    mu = 0;
    sigma = 2;
    w = normrnd(mu, sigma, 1, n_el);
    freq = datasample(1:n_el*3, n_el, 'Replace', false);

    %f = chebfun(@(x)0.05*sin(300*x), dom);
    % To remove wiggling, use the following
    f = chebfun(@(x)0, dom);
    for i = 1:n_el
        f = f + chebfun(@(x)w(i)*sin(freq(i)*x), dom);
    end
    f = 1/n_el*f;
end

function f = rand_cheb(dom)
    n_el = 10;
    mu = 0;
    sigma = 2;
    w = normrnd(mu, sigma, 1, n_el);
    
    f = chebfun(@(x)0, dom);
    for i = 1:n_el
        f = f + w(i)*chebpoly(i, dom);
    end
    f = 1/n_el*f;
end

function f = rand_gaus(dom, lambda)
    domain_length = dom(2) - dom(1);
    K = chebfun2(@(x,y)exp(-(x-y).^2/(2*domain_length^2*lambda^2)), [dom,dom]);
    L = chol(K, 'lower');
    u = randn(rank(L),1);
    f = L*u;
end
