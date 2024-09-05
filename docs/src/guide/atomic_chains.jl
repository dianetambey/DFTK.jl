# # Modelling atomic chains
#
# In [Periodic problems and plane-wave discretisations](@ref periodic-problems) we already
# summarised the net effect of Bloch's theorem.
# In this notebook, we will explore some basic facts about periodic systems,
# starting from the very simplest model, a tight-binding monoatomic chain.
# The solutions to the hands-on exercises are given at the bottom of the page.

# ## Monoatomic chain
#
# In this model, each site of an infinite 1D chain is a degree of freedom, and
# the Hilbert space is $\ell^2(\mathbb Z)$, the space of square-summable
# biinfinite sequences $(\psi_n)_{n \in \mathbb Z}$.
#
# Each site interacts by a "hopping term" with its neighbors, and the
# Hamiltonian is
# ```math
# H = \left(\begin{array}{ccccc}
#   \dots&\dots&\dots&\dots&\dots \\
#   \dots& 0 & 1 & 0 & \dots\\
#   \dots&1 & 0 &1&\dots \\
#   \dots&0 & 1 & 0& \dots  \\
#   \dots&\dots&\dots&\dots&…
# \end{array}\right)
# ```
#
# !!! tip "Exercise 1"
#     Find the eigenstates and eigenvalues of this Hamiltonian for a finite system of 
#     $N$ sites with periodic boundary counditions.
#
# !!! tip "Exercise 2"
#     Do the same when the system is more general.
#     (solve the second-order recurrence relation).
#
# We are now going to code this:

function build_monoatomic_hamiltonian(N::Integer, t)
    H = zeros(N, N)
    for n = 1:N-1
        H[n, n+1] = H[n+1, n] = t
    end
    H[1, N] = H[N, 1] = t  # Periodic boundary conditions
    H
end

# !!! tip "Exercise 3"
#     Compute the eigenvalues and eigenvectors of this Hamiltonian.
#     Plot them, and check whether they agree with theory.

# ## Diatomic chain
# Now we are going to consider a diatomic chain `A B A B ...`, where the coupling
# `A<->B` ($t_1$) is different from the coupling `B<->A` ($t_2$). We will use a new
# index $\alpha$ to denote the `A` and `B` sites, so that wavefunctions are now
# sequences $(\psi_{\alpha n})_{\alpha \in \{1, 2\}, n \in \mathbb Z}$.
#
# !!! tip "Exercise 4"
#     Show that eigenstates of this system can be looked for in the form
#     ```math
#        \psi_{\alpha k n} = u_{\alpha} e^{ikn}
#     ```
#
# !!! tip "Exercise 5"
#     Show that, if $\psi$ is of the form above
#     ```math
#        (H \psi)_{\alpha n} = (H_k u)_\alpha e^{ikn},
#     ```
#     where
#     ```math
#     H_k = \left(\begin{array}{cc}
#     0                & t_1 + t_2 e^{-ik}\\
#     t_1 + t_2 e^{ik} & 0
#     \end{array}\right)
#     ```
#
# Let's now check all this numerically:

function build_diatomic_hamiltonian(N::Integer, t1, t2)
    ## Build diatomic Hamiltonian with the two couplings
    ## ... <-t2->   A <-t1-> B <-t2->   A <-t1-> B <-t2->   ...
    ## We introduce unit cells as such:
    ## ... <-t2-> | A <-t1-> B <-t2-> | A <-t1-> B <-t2-> | ...
    ## Thus within a cell the A<->B coupling is t1 and across cell boundaries t2

    H = zeros(2, N, 2, N)
    A, B = 1, 2
    for n = 1:N
        H[A, n, B, n] = H[B, n, A, n] = t1  # Coupling within cell
    end
    for n = 1:N-1
        H[B, n, A, n+1] = H[A, n+1, B, n] = t2  # Coupling across cells
    end
    H[A, 1, B, N] = H[B, N, A, 1] = t2  # Periodic BCs (A in cell1 with B in cell N)
    reshape(H, 2N, 2N)
end

function build_diatomic_Hk(k::Integer, t1, t2)
    ## Returns Hk such that H (u e^ikn) = (Hk u) e^ikn
    ##
    ## intra-cell AB hopping of t1, plus inter-cell hopping t2 between
    ## site B (no phase shift) and site A (phase shift e^ik)
    [0                 t1 + t2*exp(-im*k);
     t1 + t2*exp(im*k) 0                 ]
end

using Plots
function plot_wavefunction(ψ)
    p = plot(real(ψ[1:end]), label="Re A")
    plot!(p, real(ψ[2:end]), label="Re B")
end

# !!! tip "Exercise 6"
#     Check the above assertions. Use a $k$ of the form
#     $2 π \frac{l}{N}$ in order to have a $\psi$ that has the periodicity
#     of the supercell ($N$).

# !!! tip "Exercise 7"
#     Plot the band structure, i.e. the eigenvalues of $H_k$ as a function of $k$
#     Use the function `build_diatomic_Hk` to build the Hamiltonians.
#     Compare with the eigenvalues of the ("supercell") Hamiltonian from
#     `build_diatomic_hamiltonian`. In the case $t_1 = t_2$, how do the bands follow
#     from the previous study of the monoatomic chain?

# !!! tip "Exercise 8"
#     Repeat the above analysis in the case of a finite-difference
#     discretization of a continuous Hamiltonian $H = - \frac 1 2 \Delta + V(x)$
#     where $V$ is periodic
#     *Hint:* It is advisable to work through [Comparing discretization techniques](@ref)
#     before tackling this question.

# ## Solutions
#
# ### Exercise 1
# $\psi_n$ is periodic over ``[1, N]``, therefore a basis of plane waves $e_k(x) = e^{i k⋅x}$ can be used with ``k = 2πl / N (l [1, N])``
# ``E_{n k} c_{k n} e^{imk n} = c_{k (n - 1)} e^{imk (n - 1)} + c_{k (n + 1)} e^{imk (n + 1)}``
# Therefore, the eigenvalues are: $E_{n k} = 2 cos(k) \frac {c_{k (n + 1)}} {c_{k n}}$
# And the eigenstates: $\psi_n = \sum_{k}{} c_kn e^{im{k n}} = \sum_{k}{} \psi_n,k$
#
# ### Exercise 2
# To approximate the general case, one can take the finite problem in the limit that N goes to infinity.
# all k would then go to zero, and there will be a second-order recurrence relation: $E_n c_n = c_{n - 1} + c_{n + 1}$
# The absolute value of the coefficients will then be equal and,
# there will be 2 solutions for the energy: $E_n = \pm 2 as c_n = \pm c_{n+1}$ 
# There will an infinite of eigenstates which are the infinite vectors with the corresponding coordinates $(-1)^n \lambda \in {\mathbb{R}}\exp\ast$ and $\lambda \in {\mathbb{R}}\exp\ast$, and their eigenvalues will be (respectfully) -2 and 2.
#
# ### Exercise 3
# We can create the matrix using $N = 10$ and $t = 1$ and then get the eigenstates and eigenvalues:
Ev , Es = eigen(build_monoatomic_hamiltonian(10, 1))
plot(Ev)
# Then we can increase N to 1000 to slowly approach the limiting case:
Evl , Esl = eigen(build_monoatomic_hamiltonian(1000, 1))
plot(Evl)
# The values become more continuous and tend to look like the following graph:
 ![eigenvalues of the limiting case](blob:http://localhost:1234/688f4102-5df7-4f56-a5bd-f60f0e895c43)
#
# ### Exercise 4
# The diatomic chain represents a periodic system that is defined by N A-B pairs. Therefore the eigenstates from exercice 1 can be used as a template.
# The difference here is that in a cell there are 2 different atoms and thus 2 different couplings, therefore a coefficient has to be added to differentiate the 2 cases.
# We then get the corresponding form.
#
# ### Exercise 5
# We can start by looking at a small sample to visualise the situation. Take 3 pairs of A-B : 1A-B_A-B_A-B1 (where 1 indicates that A and B are connected).
# The best way to define a unit cell is to consider the pair A-B.
# We can see that if we look at A (``u_1``) in cell n then the 2 neighbors are respectfully B from the previous cell (``t_2`` coupling from cell ``n - 1``) and B from the identical cell (``t_1`` coupling from cell ``n``).
# if we look at B (``u_2``) then the 2 neighbors are respectfully A from the same cell (``t_1``, ``n``) and A from the next cell (``t_2``, ``n + 1``).
# We can then write the result by applying hamiltonian and find the analogous result:
# ```math
#     H ψ_{\alpha k n} = \left(\begin{array}{cc} H u_1 e^{i k⋅n} \\ H u_2 e^{i k⋅n}\right
#     = \left(\begin{array}{cc} t_1 e^{i k⋅n} + t_2 e^{i k⋅(n-1)} \\ t_1 e^{i k⋅n} + t_2 e^{i k⋅(n+1)}\right
#     = e^{i k⋅n)}  \left(\begin{array}{cc} 0 & t_1 + t_2 e^{-i k} \\ 0 & t_1 + t_2 e^{i k}\right_\alpha
# ```
# where \alpha denotes the matrix line number and atom in the cell.
# ### Exercise 6
# To check the above statements, one can get the eigenvalues amd eigenstates of ``H_k`` for a certain ``k`` (here $k=10$)
eigen(build_diatomic_Hk(10, 0.7, 0.3))
#
# ### Exercise 7
# - To plot the values of ``H_k`` in function of ``k`` we can first create 2 functions to get each eigenvalue of ``H_k``:
function frange1(k)
    L, V = eigen(build_diatomic_Hk(k, 0.7, 0.3))
    L[1]
end

function frange2(k)
    L, V = eigen(build_diatomic_Hk(k, 0.7, 0.3))
    L[2]
end
# Then we can define an ``k`` range and plot the eigenvalues:
krange = 10:5:100
p1 = plot(krange, frange1.(krange); legend=false)
p1 = plot(krange, frange2.(krange); legend=false)
plot!(p1 , p2)
# The eigenvalues are opposite for $\alpha=1, 2$ which is logical since for each case the atoms are surrounded by the same couplings but in opposite direction.
# - We can plot the eigenvalues of each hamiltonian to compare
# Eigenvectors and eigenvalues for H:
Ed , Vd = eigen(build_diatomic_hamiltonian(5, 0.7, 0.3))
# Plot for ``H``:
plot_wavefunction(Ed)
# Eigenvectors and eigenvalues for ``H_k``:
Ek , Vk = eigen(build_diatomic_Hk(5,0.7,0.3))
# Plot for ``H_k``:
plot_wavefunction(Ek)
# - If ``t_1 = t_2``: 
# the 2 couplings are identical which means that it corresponds to the exact same case as exercice 1. The only difference is the atoms.
#
# ### Exercise 8
# 

