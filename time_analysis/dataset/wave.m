% This program describes a moving 1-D wave
% using the finite difference method
% It is based on the code from Praveen Ranganath : 
% https://www.mathworks.com/matlabcentral/fileexchange/43001-1d-wave-propagation-a-finite-difference-approach

clc
close all;
clear all;

% Initialization

% Select forcing type and boundary condition type
forcing  = 'pwl';  % choose from cheb sine pwl gaus
boundary = 'space'; % either time or space for the boundary conditions

Nx = 101;     
dx = 1/(Nx-1);
x(:,1) = (0:Nx-1)*dx;

mpx = (Nx+1)/2; % Mid point of x axis
                % ( Mid pt of 1 to 101 = 51 here )
                
T = 401;       % Total number of time steps
dt = 0.001;
t(:,1)= (0:T-1)*dt;

v = 1;        % Wave velocity
c = v*(dt/dx);  % CFL condition

dom = [0,1];
Nsample = 200;

% Arrays to store the data
U = zeros(101, 101, Nsample);
F = zeros(101, Nsample);

for k=1:Nsample

    u = zeros(T,Nx);  % U(x,t) = U(space,time)

    % Sample the random forcing that determines the boundary conditions
    if strcmp(forcing, 'pwl')
        f = rand_pwl(4, dom);
    elseif strcmp(forcing, 'sine')
        f = rand_sine(dom);
    elseif strcmp(forcing, 'cheb')
        f = rand_cheb(dom);
    elseif strcmp(forcing, 'gaus')
        f = rand_gaus(dom, 0.03);
    end
    
    % Initial condition
    if strcmp(boundary, 'time')
        u(:,1) = f(t);
        u(:,2) = f(t);

        % For filename
        bd = '_t';
        
    elseif strcmp(boundary, 'space')
        u(1, :) = f(x);
        u(2, 1) = u(1, 1) - (1/2)*c*c*(u(1, 2)-2*u(1, 1));
        u(2, end) = u(1, end) - (1/2)*c*c*(u(1, end-1)-2*u(1, end));
        for j=3:Nx-1
            u(2, j) = u(1, j) - (1/2)*c*c*(u(1, j+1)-2*u(1, j)+u(1, j-1));
        end

        % For filename
        bd = '_x';
    end
    
    % Finite Difference Scheme
    for j = 3:T
        for i = 2:Nx-1
            U1 = 2*u(j-1,i)-u(j-2,i);
            U2 = u(j-1,i-1)-2*u(j-1,i)+u(j-1,i+1);
            u(j,i) = U1 + c*c.*U2;    
        end                   
    end
    
    % We save the results
    u = downsample(u, 4);
    U(:,:,k) = u;

    if strcmp(boundary, 'time')
        F(:,k) = u(:,1);
    elseif strcmp(boundary, 'space')
        F(:, k) = u(1, :);
    end
end

fX = x;
U = reshape(U, 101*101, Nsample);

% We write the .mat file
save(sprintf("wave_" + forcing + bd + ".mat"),"U","F","fX")

function pwlf = rand_pwl(n, dom)
% Generates a continuous piecewise linear function of n pieces
% with uniformly sampled nodes
% with slope sampled from a Gaussian distribution

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
    n_el = 5;
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
    n_el = 5;
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
