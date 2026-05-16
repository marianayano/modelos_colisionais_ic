import numpy as np
import sympy

I        = sympy.I
one      = 0*I + 1
rho_vars = []
tags     = []
numerical_types = (int, float, complex, np.integer, np.floating, np.complexfloating)

def help(query=""):
    if not isinstance(query, str):
        print("Queries must be strings, remember to use quotation marks. Run help() with no arguments for more information")
        return
    functions = ['build_rho', 'coherent_entry', 'commutator', 'commutator_T', 'dagger', 'dagger_T', 'dissipator_entry', 'hermitian2vec', 'embed_A', 'embed_B', 'id_n', 'make_odeint_function', 'partial_trace_A', 'partial_trace_B', 'product_T', 'simple_expand', 'simple_simplify', 'tensor_product', 'trace', 'vec2hermitian']
    constants = ['I', 'ident', 'one',  'sm', 'sp', 'sx', 'sy', 'sz']
    if query not in functions + constants:
        print('You can get documentation of the functions and constants by running help("name")')
        print("\nImplemented functions:\n\t" + "\n\t".join(functions))
        print("\n\nImplemented constants:\n\t" + "\n\t".join(constants))
        return
    if query in functions:
        print(eval(query + ".__doc__"))
    else:
        match query:
            case "I":
                print("Imaginary unit (sympy symbol), I**2 == -1")
            case "ident":
                print("2x2 identity matrix (sympy symbol). Equivalent to id_n(2)")
            case "one":
                print("Real unit (sympy.symbol). Can be used to multiply an integer or a matrix/array of integers and convert it to a symbolic integer/matrix")
            case "sm":
                print("The 𝞼_+ matrix (np.array of sympy.symbol)")
            case "sp":
                print("The 𝞼_- matrix (np.array of sympy.symbol)")
            case "sx":
                print("The 𝞼_x matrix (np.array of sympy.symbol)")
            case "sy":
                print("The 𝞼_y matrix (np.array of sympy.symbol)")
            case "sz":
                print("The 𝞼_z matrix (np.array of sympy.symbol)")

##############################################################################

def tensor_product(a, b):
    """tensor_product(A, B):
\tA: NxN 2d array (either nested list or np.array)
\tB: MxM 2d array (either nested list or np.array)
\toutputs A ⭙ B: NxMxNxM 4d array (np.array)

Takes two matrixes/operators A and B with coordinates A_{i,j} and B_{k,l} and returns A ⭙ B with coordinates [A ⭙ B]_{(i,k), (j,l)}

We assume that A[i][j] == A_{i,j}, B[k][l] = B_{k,l} for the entries
For the output, tensor_product(A, B)[i][k][j][l] == [A ⭙ B]_{(i,k), (j,l)}"""
    # assuming a and b are square matrices
    N = len(a)
    M = len(b)
    prod = [[[[a[i][j]*b[k][l] for l in range(M)] for j in range(N)] for k in range(M)] for i in range(N)]
    return np.array(prod)

def partial_trace_A(T):
    """partial_trace_A(T):
\tT: NxMxNxM 4d array (either nested list or np.array)
\toutputs: Tr_A(T): MxM 2d array (np.array)

Takes an operator T with coordinates T_{(i,k), (j,l)} in a product space A ⭙ B and returns the partial trace 𝙏=Tr_A(T) over the first space, with
𝙏_{k,l} = 𝝨_i T_{(i,k), (i,l)}

We assume that that T[i][k][j][l] == T_{(i,k), (j,l)} for the entry
For the output, partial_trace_A(T)[k][l] == 𝙏_{k,l}"""
    N = len(T)
    M = len(T[0])
    trace = [[sum([T[i][k][i][l] for i in range(N)]) for l in range(M)] for k in range(M)]
    return np.array(trace)

def partial_trace_B(T):
    """partial_trace_B(T):
\tT: NxMxNxM 4d array (either nested list or np.array)
\toutputs: Tr_B(T): NxN 2d array (np.array)

Takes an operator T with coordinates T_{(i,k), (j,l)} in a product space A ⭙ B and returns the partial trace 𝙏=Tr_B(T) over the second space, with
𝙏_{i,j} = 𝝨_k T_{(i,k), (j,k)}

We assume that that T[i][k][j][l] == T_{(i,k), (j,l)} for the entry
For the output, partial_trace_B(T)[i][j] == 𝙏_{i,j}"""
    N = len(T)
    M = len(T[0])
    trace = [[sum([T[i][k][j][k] for k in range(M)]) for j in range(N)] for i in range(N)]
    return np.array(trace)

def trace(M):
    """trace(M):
\tM: NxN 2d array (either nested list or np.array)
\toutputs: Tr(M): number

Takes an operator M with coordinates M_{i, j} in some space A and returns the trace 𝙏=Tr(M), with 𝙏 = 𝝨_i M_{i, i}

We assume that that M[i][j] == M_{i, j} for the entry
"""
    N = len(M)
    return sum([M[i][i] for i in range(N)])

def product_T(A, B):
    """product_T(A, B):
\tA: NxMxNxM 4d array (either nested list or np.array)
\tB: NxMxNxM 4d array (either nested list or np.array)
\toutputs: A.B: NxMxNxM 4d array (np.array)

Takes two operators A and B with coordinates A_{(i,k), (j,l)} and B_{(i,k), (j,l)} in some product space U ⭙ V and returns the product/composition
A.B of the two operators, that is, the operator of U ⭙ V with coordinates
[A.B]_{(i,k), (j,l)} = 𝝨_{n,m}  A_{(i,k), (n,m)}.B_{(n,m), (j,l)}

We assume that that A[i][k][j][l] == A_{(i,k), (j,l)} and B[i][k][j][l] == B_{(i,k), (j,l)} for the entries
For the output, product_T(A, B)[i][k][j][l] == [A.B]_{(i,k), (j,l)}"""
    #assuming A and B are in the same product space
    N = len(A)
    M = len(A[0])
    prod = [[[[sum([A[i][a][k][c]*B[k][c][j][b] for k in range(N) for c in range(M)]) for b in range(M)] for j in range(N)] for a in range(M)] for i in range(N)]
    return np.array(prod)

def embed_A(B, n):
    """embed_A(T, N):
\tT: MxM 2d array (either nested list or np.array)
\tN: integer
\toutputs: 𝕀 ⭙ T: NxMxNxM 4d array (np.array)

Takes an operator T with coordinates T_{i, j} in some space B and an integer N and returns the same operator in a space A ⭙ B, where dim(A)=N. That is,
we return 𝕀_N ⭙ T, where 𝕀_N is the NxN identity matrix

We assume that that T[i][j] == T_{i, j}
This is just a short hand, using tensor_product internally"""
    return tensor_product(np.eye(n, dtype = int), B)

def embed_B(A, n):
    """embed_B(T, M):
\tT: NxN 2d array (either nested list or np.array)
\tM: integer
\toutputs: T ⭙ 𝕀: NxMxNxM 4d array (np.array)

Takes an operator T with coordinates T_{i, j} in some space A and an integer M and returns the same operator in a space A ⭙ B, where dim(B)=M. That is,
we return T ⭙ 𝕀_M, where 𝕀_M is the MxM identity matrix

We assume that that T[i][j] == T_{i, j}
This is just a short hand, using tensor_product internally"""
    return tensor_product(A, np.eye(n, dtype = int))

def id_n(n):
    """id_n(n):
\tn: integer
\toutputs: 𝕀_n: nxn 2d array (np.array of sympy.symbol)

Takes an integer n and returns a symbolic version of 𝕀_n, the nxn identity"""
    return np.eye(n, dtype = int)*one

def commutator_T(A, B):
    """commutator_T(A, B):
\tA: NxMxNxM 4d array (either nested list or np.array)
\tB: NxMxNxM 4d array (either nested list or np.array)
\toutputs: [A,B]: NxMxNxM 4d array (np.array)

Takes two operators A and B with coordinates A_{(i,k), (j,l)} and B_{(i,k), (j,l)} in some product space U ⭙ V and returns their commutator
[A,B] = A.B - B.A

We assume that that A[i][k][j][l] == A_{(i,k), (j,l)} and B[i][k][j][l] == B_{(i,k), (j,l)} for the entries
For the output, commutator_T(A, B)[i][k][j][l] == [A,B]_{(i,k), (j,l)}

This is just a short hand, using product_T internally"""
    #assuming A and B are in the same product space
    return product_T(A, B) - product_T(B, A)

def commutator(A, B):
    """commutator(A, B):
\tA: NxN 2d array (either nested list or np.array)
\tB: NxN 2d array (either nested list or np.array)
\toutputs: [A,B]: NxN 2d array (np.array)

Takes two operators A and B with coordinates A_{i, j} and B_{i, j} in some space U and returns their commutator
[A,B] = A.B - B.A

We assume that that A[i][j] == A_{i, j} and B[i][j] == B_{i, j} for the entries
For the output, commutator(A, B)[i][j] == [A,B]_{i, j}"""
    return np.dot(A, B) - np.dot(B, A) 

def simple_expand(M):
    """simple_expand(M):
\tM: NxN 2d array (either nested list or np.array)
\toutputs: M: NxN 2d array (np.array)

Takes a matrix M with coordinates M_{i, j} in some space U and expands it symbolically

The expansion may lead to some simplifications, but sometimes a more involved procedure is required (simple_simplify)"""
    return np.array(sympy.Matrix(M).expand())

def simple_simplify(M):
    """simple_simplify(M):
\tM: NxN 2d array (either nested list or np.array)
\toutputs: M: NxN 2d array (np.array)

Takes a matrix M with coordinates M_{i, j} in some space U and simplifies it symbolically

Often times the program requires an expansion before simplification becomes viable (in these cases, first run simple_expand and
then simplify the result)"""
    return np.array(sympy.Matrix(M).expand())

def dagger(M):
    """dagger(M):
\tM: NxN 2d array (either nested list or np.array)
\toutputs: M^†: NxN 2d array (np.array)

Takes a matrix M with coordinates M_{i, j} in some space U and returns its complex conjugate M^† 

We assume that that M[i][j] == M_{i, j} for the entry"""
    return np.array(sympy.Matrix(M).H)

def dagger_T(M):
    """dagger(M):
\tM: NxMxNxM 4d array (either nested list or np.array of complex or sympy.symbols)
\toutputs: M^†: NxMxNxM 4d array (np.array)

Takes an operator M with coordinates M_{(i,k), (j,l)} in some space U ⭙ V and returns its complex conjugate M^† 

We assume that that M[i][k][j][l] == M_{(i,k), (j,l)} for the entry
For the output, M^†[i][k][j][l] == M_{(j,l), (i,k)}*"""
    shape = M.shape
    return np.array([x.conjugate() for x in M.reshape((-1,))]).reshape(shape).transpose((2,3,0,1))

def dissipator_entry(g, L, A, rhoS, rhoA):
    """dissipator_entry(g, L, A, rhoS, rhoA):
\tg: number
\tL: NxN 2d array (either nested list or np.array)
\tA: MxM 2d array (either nested list or np.array)
\trhoS: NxN 2d array (either nested list or np.array)
\trhoA: MxM 2d array (either nested list or np.array)
\toutputs: 𝛾_-.𝓓(L) + 𝛾_+.𝓓(L^†): NxN 2d array (np.array)

This function calculates the contribution to a Lindblad dissipator due to the jump operator L of the system and the corresponding jump operator A of the ancilla. rhoS is the system density matrix while rhoA is the thermal ancilla density matrix. g is the coupling of this interaction mode.

𝛾_- equals |g|^2.Tr(rho_A.A.A^†)
𝛾_+ equals |g|^2.Tr(rho_A.A^†.A)"""
    L_  = dagger(L)
    LL_ = np.dot(L, L_)
    L_L = np.dot(L_, L)
    A_  = dagger(A)
    AA_ = np.dot(A, A_)
    A_A = np.dot(A_, A)
    gm  = g*g.conjugate()*trace(np.dot(rhoA, AA_))
    gp  = g*g.conjugate()*trace(np.dot(rhoA, A_A))
    DL  = np.dot(np.dot(L, rhoS), L_) - (np.dot(L_L, rhoS) + np.dot(rhoS, L_L))/2
    DL_ = np.dot(np.dot(L_, rhoS), L) - (np.dot(LL_, rhoS) + np.dot(rhoS, LL_))/2
    return gm*DL + gp*DL_

def coherent_entry(g, L, A, rhoS, chiA):
    """coherent_entry(g, L, A, rhoS, chiA):
\tg: number
\tL: NxN 2d array (either nested list or np.array)
\tA: MxM 2d array (either nested list or np.array)
\trhoS: NxN 2d array (either nested list or np.array)
\tchiA: MxM 2d array (either nested list or np.array)
\toutputs: -i[G, 𝜌_S]: NxN 2d array (np.array)

This function calculates the contribution to the coherent contribution -i[G, 𝜌_S] (with 𝜆 absorbed in 𝜒) to the Lindblad dissipator due to the jump operator L of the system and the corresponding jump operator A of the ancilla. rhoS is the system density matrix while chiA is the coherent perturbation of the thermal ancilla density matrix (𝜒). g is the coupling of this interaction mode."""
    avA  = trace(np.dot(chiA, A))
    avA_ = trace(np.dot(chiA, dagger(A)))
    G_AL = g*avA*dagger(L) + g.conjugate()*avA_*L
    return -I*commutator(G_AL, rhoS)

def build_rho(n, tag = ""):
    """build_rho(n, tag = ""):
\tn: integer
\ttag: string (optional argument, "" by default)
\toutputs: 𝜌: nxn 2d array (np.array of sympy.symbols)
\tside effect: updates rho_vars (if a tag is reused rho_Vars is reset before being updated)

This function takes an integer n and returns a symbolic density matrix. tag is a string that can be used to distinguish the elements of different 𝜌 if needed

The symbols used are <tag>Ri_j for Re(𝜌_{i,j}) and <tag>Ji_j for Im(𝜌_{i,j})"""
    global rho_vars
    def name(a, b):
        if a > b:
            return "J" + str(b) + "_" + str(a)
        else:
            return "R" + str(a) + "_" + str(b)
    def compose(R, a, b):
        if a < b:
            return R[a][b] + I*R[b][a]
        if a > b:
            return R[b][a] - I*R[a][b]
        return R[a][a]
    symbols = np.array([[sympy.symbols(tag + name(a,b), real=True) for b in range(n)] for a in range(n)])
    rhoS    = np.array([[compose(symbols,a,b) for b in range(n)] for a in range(n)])
#    return rhoS, symbols.reshape((-1,))
    symbols = list(symbols.reshape((-1,)))
    if tag in tags:
        rho_vars = symbols
    else:
        rho_vars += symbols
        tags.append(tag)
    return rhoS

def hermitian2vec(M):
    """hermitian2vec(M):
\tM: hermitian matrix (nested list or np.array)
\toutputs: vec(M): 1d array (np.array)

This function takes an hermitian matrix M and returns a vector containing all the information to reconstruct it. This can be thought of as finding the coordinates of M in the (real) space of hermitian matrices

The main purpose of this function is to create a consitent vector representation for use together with scipy.integrate.odeint (using this representation also guarantees hermiticity won't be violated by numerical errors)"""
    numerical = isinstance(M[0][0], numerical_types) 
    n = len(M)
    def extract(M, pos):
        a = pos // n
        b = pos % n
        if a > b:
            retval = sympy.im(M[b][a])
        else:
            retval = sympy.re(M[a][b])
        if numerical:
            retval = float(retval.evalf())
        return retval
    return np.array([extract(M, pos) for pos in range(n**2)])

def vec2hermitian(v):
    """vec2hermitian(M):
\tvec(M): 1d array (list or np.array) [with n^2 elements]
\toutputs: M: 2d array (np.array)

This function reads a vector representation (using the same convention as hermitian2vec) and returns the represented hermitian matrix

The main purpose of this function is to take the vectors returned by scipy.integrate.odeint and convert them back into density matrices in order to make finding averages more convenient"""
    numerical = isinstance(v[0], numerical_types)
    n = int(len(v)**.5)
    def combine(v, a, b):
        if numerical:
            i = 1.j
        else:
            i = I
        if a < b:
            return v[a*n + b] + i*v[b*n + a]
        if a > b:
            return v[b*n + a] - i*v[a*n + b]
        return v[a*n + a]
    return np.array([[combine(v, a, b) for b in range(n)] for a in range(n)])

def make_odeint_function(drho, subs = dict()):
    """make_odeint_function(drho_vec, subs = dict()):
\tdrho: 2d array (either nested list or np.array)
\tsubs: dictionary (optional)

Take the derivative d𝜌/dt and a dictionary of substitutions and return a function that can be used by scipy.integrate.odeint (uses the f(y,t) convention -- see the documentation of odeint)"""
    drho_vec = hermitian2vec(drho)
    lambdified = sympy.lambdify(rho_vars, [der.subs(subs) for der in drho_vec], "numpy")
    return eval("lambda y, t : lambdified(" + ",".join(["y[" + str(i) + "]" for i in range(len(drho_vec))]) + ")", locals()) 

##############################################################################

sx    = np.array([[0, 1], [1, 0]])*one
sy    = np.array([[0, -I], [I, 0]])
sz    = np.array([[1, 0], [0, -1]])*one
sm    = np.array([[0, 0], [1, 0]])*one
sp    = np.array([[0, 1], [0, 0]])*one
ident = np.array([[1, 0], [0, 1]])*one

