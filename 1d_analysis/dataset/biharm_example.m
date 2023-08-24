function biharm_example(varargin)
    % This file is based on the code of John Burkardt distributed under the GNU LGPL license and the one of Nicolas Boulle
    
    % We solve the 1d biharmonic equation in domain dom with boundary conditions
    % ul ur upl upr on the values of the solution and its first derivative on the left and right boubndary of dom

    % Definition of domain
    dom = [0,3];

    % Evaluation points for G
    NGx = 1000;
    NGy = 1000;
    XG = linspace(dom(1), dom(2), NGx)';
    YG = linspace(dom(1), dom(2), NGy)';

    % Number of sampled functions f
    Nsample = 100;

    % Noise level of the solutions u
    % If we wish to add noise to the boundary
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
        f = rand_pwl(4, dom);
        x = linspace(dom(1), dom(2), n);
        x = x';
        f_v = f(x);
        u = biharmonic_fd1d(n, f_v, dom);

        % Evaluate at the training points
        U(:,i,:) = u;
        F(:,i,:) = f(Y);
    end

    % Compute homogeneous solution
    u_hom = biharmonic_fd1d(n, zeros(n, 1), dom);
    U_hom = u_hom;

    % Add Gaussian noise to the solution
    U = U.*(1 + noise_level*randn(size(U)));

    save(sprintf('biharm_pwl.mat'),"X","Y","U","F","U_hom","XG","YG")
end

function u = biharmonic_fd1d(n, f_v, dom)
    % solves the 1d biharmonic equation using finite differences.
  
    % Set the grid.
    xmin = dom(1);
    xmax = dom(2);
    x = linspace ( xmin, xmax, n );
    x = x';
    dx = ( xmax - xmin ) / ( n - 1 );
  
    % Get the basic matrix
    A = biharmonic_matrix ( n );
    
    % Get the basic right hand side.
    b = f_v;
    b = b * dx^4;
    
    % Get the boundary conditions on the left and right endpoints for U and U'.
    ul = 2.0;
    ur = 1.0;
    upl = 3.0;
    upr = 4.0;

    % Modify A and b for the Dirichlet condition on U at left.
    A(1,1) = 1.0;
    A(1,2) = 0.0;
    A(1,3) = 0.0;
    b(1) = ul;
    
    % Modify A and b for the Dirichlet condition on U at right.
    A(n,n-2) = 0.0;
    A(n,n-1) = 0.0;
    A(n,n)   = 1.0;
    b(n) = ur;

    % Modify A and b for the Dirichlet condition on U' at left.
    A(2,2) = 7.0;
    b(2) = b(2) + 2.0 * dx * upl;

    % Modify A and b for the Dirichlet condition on U' at right.
    A(n-1,n-1) = 7.0;
    b(n-1) = b(n-1) - 2.0 * dx * upr;
  
    % Solve the system.
    u = A \ b;
end

function A = biharmonic_matrix ( n )
    % returns a matrix for the 1D biharmonic equation.
    % To compute the fourth derivative, the matrix entries would normally
    % include a divisor of dx^4.  For simplicity, this divisor is omitted.
    % To solve A*x=b correctly, then, the right hand side b must be
    % multiplied by dx^4.
    %
    % The first two and last two rows of A and b must be modified, to account
    % for boundary conditions.

    e = ones ( n, 1 );
    B = [1,-4,6,-4,1];
    eB = e*B;
  
    A = spdiags ( eB, -2:+2, n, n );   
end

function pwlf = rand_pwl(n, dom)
    % Generates a random continuous piecewise linear function of n pieces
    % with uniformly sampled nodes
    % with slopes sampled from gaussian distribution
    
    % Define the range of the function as dom and sample n+1 points uniformly
    % in the domain
    x = unifrnd(dom(1), dom(2), n+1, 1);
    x = sort(x);
    
    % Generate random coefficients for the function
    sigma = 3;
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
    % Generate a random function sampled as an average of sine functions with gaussian weights

    n_el = 10;
    mu = 0;
    sigma = 2;
    w = normrnd(mu, sigma, 1, n_el);
    freq = datasample(1:n_el*3, n_el, 'Replace', false);

    f = chebfun(@(x)0, dom);
    for i = 1:n_el
        f = f + chebfun(@(x)w(i)*sin(freq(i)*x), dom);
    end
    f = f/n_el
end

function f = rand_cheb(dom)
    % Generate a random function sampled as an average of Chebyshev polynomials with gaussian weights
    
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
