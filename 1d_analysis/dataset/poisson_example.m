function poisson_example(varargin)
    % Solve the 1d Poisson equation for Nsample random forcings in dom with Dirichlet boundary conditions

    % forcing type
    forcing = "pwl";

    % Definition of domain
    dom = [0,1];

    % Number of sampled functions f
    Nsample = 100;

    % Noise level of the solutions u
    noise_level = 0;

    % Training points for f
    Nf = 200;
    Y = linspace(dom(1), dom(2), Nf)';

    % Training points for u
    Nu = 100;
    X = linspace(dom(1), dom(2), Nu)';

    % Number of points for the solver
    n = Nu;

    % Initialize data arrays
    U = zeros(Nu, Nsample, 1);
    F = zeros(Nf, Nsample, 1);

    for i = 1:Nsample
        if strcmp(forcing, 'pwl')
            f = rand_pwl(4, dom);
        elseif strcmp(forcing, 'sine')
            f = rand_sin(dom);
        elseif strcmp(forcing, 'cheb')
            f = rand_cheb(dom);
        elseif strcmp(forcing, 'gaus')
            f = rand_gaus(dom, 0.03);
        end

        x = linspace(dom(1), dom(2), n);
        x = x';
        f_v = f(x);
        u = poisson_fd1d(n, f_v, dom);

        % Evaluate at the training points
        U(:,i,:) = u;
        F(:,i,:) = f(Y);
    end

    % Compute homogeneous solution
    u_hom = poisson_fd1d(n, zeros(n, 1), dom);
    U_hom = u_hom;

    % Add Gaussian noise to the solution
    U = U.*(1 + noise_level*randn(size(U)));

    save(sprintf('poisson_' + forcing + '_test.mat'),"X","Y","U","F","U_hom")
end

function u = poisson_fd1d(n, f_v, dom)
    % Solves the Poisson equation with finite differences

    % Set the grid.
    xmin = dom(1);
    xmax = dom(2);
  
    dx = ( xmax - xmin ) / ( n - 1 );

    % Get the basic matrix.
    A = poisson_matrix ( n );

    % Get the basic right hand side.
    b = f_v;
    b = b * dx^2;

    % Get the boundary conditions on the left and right endpoints for U and U'.
    ul = 0.0;
    ur = 0.0;

    % Modify A and b for the Dirichlet condition on U at left.
    A(1,1) = 1.0;
    A(1,2) = 0.0;
    A(1,3) = 0.0;
    b(1) = ul;

    % Modify A and b for the Dirichlet condition on U at right
    A(n,n-2) = 0.0;
    A(n,n-1) = 0.0;
    A(n,n)   = 1.0;
    b(n) = ur;

    % Solve the system.
    u = A \ b;
end

function A = poisson_matrix ( n )

  e = ones ( n, 1 );
  B = [-1,2,-1];
  eB = e*B;

  A = spdiags ( eB, -1:+1, n, n );
end

function pwlf = rand_pwl(n, dom)
    % Generates a continuous piecewise linear function of n pieces
    % with uniformly sampled nodes
    % with slope sampled form a Gaussian distribution
    
    % Define the range of the function as dom and sample n+1 points uniformly
    % in the domain
    x = unifrnd(dom(1), dom(2), n+1, 1);
    x = sort(x);
    
    % Generate random coefficients for the function
    sigma = 0;
    mu = 2;
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


function f = rand_sin(dom)
    % Generate a random function sampled as an average of sine functions with Gaussian weights

    n_el = 10;
    mu = 0;
    sigma = 2;
    w = normrnd(mu, sigma, 1, n_el);
    %freq = datasample(1:n_el*3, n_el, 'Replace', false);

    %f = chebfun(@(x)0.05*sin(300*x), dom);
    % To remove wiggling, use the following
    f = chebfun(@(x)0, dom);
    for i = 1:n_el
        f = f + chebfun(@(x)w(i)*sin(i*x), dom);
    end
    f = f/n_el
end

function f = rand_cheb(dom)
    % Generate a random function sampled as an average of Chebyshev polynomials with Gaussian weights

    n_el = 10;
    mu = 0;
    sigma = 2;
    w = normrnd(mu, sigma, 1, n_el);
    
    f = chebfun(@(x)0, dom);
    for i = 1:n_el
        f = f + w(i)*chebpoly(i, dom);
    end
    f = f/n_el
end

function f = rand_gaus(dom, lambda)
    % Generate a random function sampled from a Gaussian process with square exponential kernel K

    domain_length = dom(2) - dom(1);
    K = chebfun2(@(x,y)exp(-(x-y).^2/(2*domain_length^2*lambda^2)), [dom,dom]);
    L = chol(K, 'lower');
    u = randn(rank(L),1);
    f = L*u;
end
