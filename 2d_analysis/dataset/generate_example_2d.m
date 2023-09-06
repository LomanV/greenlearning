function generate_example_2d(varargin)

    % Make experiment reproducible
    rng(1)

    % Choose equation from poisson or biharmonic
    eq = "poisson";
    % Choose forcing type from pwl (continuous piecewise-linear) sine
    % cheb (Chebyshev polynomials) or gaus (drawn from Gaussian process)
    forcing = "gaus";
    points  = 100;

    % Definition of a square domain [0, 1]^2
    dom  = [0 1];

    % Number of sampled functions f for training
    Nsample = 200;

    % Noise level of the solutions u
    noise_level   = 0;

    % Training points for f
    Nf = points;
    fX = linspace(dom(1), dom(2), Nf);

    % Training points for u
    Nu = points;

    % Number of points for the solver
    n = points;

    % Initialize data arrays
    U = zeros(Nu*Nu, Nsample, 1);
    F = zeros(Nf*Nf, Nsample, 1);

    [x, y] = ndgrid(fX, fX);
    for i = 1:Nsample

        sprintf("Training set : i = %d/%d",i, Nsample)

        if forcing == "pwl"
            f = rand_pwl2d(4, dom);
            f_v = f(x, y);

        elseif forcing == "sine"
            f_v = rand_sin(x, y);

        elseif forcing == "cheb"
            f_v = rand_cheb(x, y);

        elseif forcing == "gaus"
            domain_length = dom(2) - dom(1);
            lambda = 0.03;
            K = chebfun2(@(x,y)exp(-(x-y).^2/(2*domain_length^2*lambda^2)), [dom,dom]);
            L = chol(K, 'lower');
            f_v = rand_gaus(L, x, y);

        else
            msg = "Unknown forcing type";
            error(msg)
        end

        
        u = fd2d(Nu, f_v, dom, eq);

        % Evaluate at the subsampled training points for u
        
        U(:, i,:) = u;
        F(:,i,:) = reshape(f_v.', [], 1);

    end

    % Compute homogeneous solution
    u_hom = fd2d(n, zeros(1, n*n), dom, eq);
    U_hom = u_hom;

    % Add Gaussian noise to the training solution
    U = U.*(1 + noise_level*randn(size(U)));

    save(sprintf(eq + '_2d_' + forcing + points + '.mat'), "fX", "U", "F", "U_hom")
end

function u = fd2d(n, f_v, dom, eq)

    % Set the grid.
    dx = ( dom(1) - dom(2) ) / ( n - 1 );
    % Get the basic matrix.
    if eq == "biharmonic"
        A = biharmonic_matrix_2d ( n );
    elseif eq == "poisson"
        A = poisson_matrix_2d(n);
    else 
        msg = 'Unvalid equation name';
        error(msg)
    end

    % Get the basic right hand side.
    b = reshape(f_v, n*n, []);
  
    if eq == "biharmonic"
        b = b .* dx^4;
    elseif eq == "poisson"
        b = b .* dx^2;
    else
        msg = "Unvalid equation name";
        error(msg)
    end

    [X, Y] = meshgrid(linspace(dom(1), dom(2), n), linspace(dom(1), dom(2), n));
  
    isDirichlet = (X==dom(1)) | (X==dom(2)) | (Y==dom(1)) | (Y==dom(2));
    b(isDirichlet) = 0.0;

    % Solve the system.
    u = A \ b;

    return
end

function A = poisson_matrix_2d ( n )

    [X, Y] = meshgrid(linspace(0, 1, n), linspace(0, 1, n));
    isDirichlet = (X==0) | (X==1) | (Y==0) | (Y==1);
    N = n*n;
    Ad = zeros(N,5);
    
    Ad(:,1) = -1; 
    Ad(:,2) = -1;
    Ad(:,4) = -1; 
    Ad(:,5) = Ad(:,1);
    
    % Use these to get center by summing the negatives
    Ad(:,3) = -sum( Ad(:,[1,2,4,5]), 2 ); % Center
    % Overwrite BC's on the A Matrix diagonals
    idx = find(isDirichlet(:)); % array of inidices of BC points
    Ad(idx,:) = 0.0; % Zero all elements of rows defined by idx
    Ad(idx,3) = 1.0; % Make center node=1
    % Create A Matrix from Ad, assigning columns to diagonals
    A = spdiags(Ad,[-n,-1,0,1,n],N,N)'; % ADDED TRANSPOSE
end

function M = biharmonic_matrix_2d ( n )
    
    % Create the 1D biharmonic matrix A
    e = ones ( n, 1 );
    B = [1,-4,6,-4,1];
    eB = e*B;

    A = spdiags ( eB, -2:+2, n, n );

    A(1,1) = 1.0;
    A(1,2) = 0.0;
    A(1,3) = 0.0;

    A(n,n-2) = 0.0;
    A(n,n-1) = 0.0;
    A(n,n)   = 1.0;
    
    % Find a way to encode the first order ux uy boundary conditions
    %A(2,2) = 7.0;
    %A(n-1,n-1) = 7.0;

    I = eye(n);

    C = [1, -2, 1];
    eC = e*C;
    E = spdiags ( eC, -1:+1, n, n );

    % Use Kronecker product to apply A to x (d^4/dx^4) to y (d^4/dy^4)
    % kron(E, E) takes care of the d^4/dx^2dy^2
    M = kron(A, I) + kron(I, A) + 2*kron(E, E);

    return
end

function pwlf = rand_pwl2d(n, dom)
    x = linspace(dom(1), dom(2), n+1);
    x = sort(x);

    y = linspace(dom(1), dom(2), n+1);
    y = sort(y);

    [x, y] = ndgrid(x,y);
    a = randn(size(x));
    pwlf = griddedInterpolant(x,y,a);
end

function z = rand_sin(x, y)
    n_el = 10;
    mu = 0;
    sigma = 2;
    w = normrnd(mu, sigma, 1, n_el);
    %freq = datasample(1:n_el*3, n_el, 'Replace', false);
    freq = randperm(n_el);

    z = zeros(size(x));
    for i = 1:n_el
        % Aligned frequencies
        %z = z + w(i)*sin(freq(i)*x).*sin(freq(i)*y);
        % Offset frequencies
        z = z + w(i)*sin(freq(i)*x).*sin(i*y);
    end
end

function z = rand_cheb(x, y)
    n_el = 10;
    mu = 0;
    sigma = 1;
    w = normrnd(mu, sigma, 1, n_el);

    z = zeros(size(x));
    shuffle = randperm(n_el);
    for i = 1:n_el
        j = shuffle(i);
        ti = ChebyshevPoly(i);
        tj = ChebyshevPoly(j);
        z = z + w(i)*polyval(ti, x).*polyval(tj, y);
    end
end

function z = rand_gaus(L, x, y)
    u = randn(rank(L),1);
    v = randn(rank(L), 1);
    f = L*u;
    g = L*v;

    z = f(x).*g(y);

end
