# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import math
import numpy as np
import numpy.linalg

FM_TO_BOHR = 1.8897259885789e-5
BOHR_TO_FM = 1.0 / FM_TO_BOHR
AU_TO_MEV = 27.211386245988e-6
MEV_TO_AU = 1.0 / AU_TO_MEV


class Options(object):

    def __init__(self):
        self.num_protons = 0
        self.num_neutrons = 0
        self.grid_size = 100
        self.rmax = 10.0 # fm
        self.lmax = 6

        self.proton_mass = 1836.15267343   # a.u.
        self.neutron_mass = 1838.68366173  # a.u.

        self.ws_V0 = 53   # MeV
        self.ws_r0 = 1.24 # fm
        self.ws_xi = 0.63 # fm

    def print(self):
        print()
        print(' options:')
        print(' number of protons (Z)   %d' % self.num_protons)
        print(' number of neutrons (N)  %d' % self.num_neutrons)
        print(' mass number (A)         %d' % (self.num_protons + self.num_neutrons))
        print(' number of grid points   %d' % self.grid_size)
        print(' integration limit (fm)  %.3f' % self.rmax)
        print()


def get_tridiagonal_matrix(N, diag_value, off_diag_value):

    A = np.zeros((N, N))

    for i in range(0, N):
        A[i,i] = diag_value

    for i in range(1, N):
        A[i - 1, i] = off_diag_value
        A[i, i - 1] = off_diag_value

    return A


def potential_coulomb_protons(options, r):

    N = options.num_neutrons
    Z = options.num_protons
    A = N + Z
    R0 = options.ws_r0 * FM_TO_BOHR * A**(1.0/3.0)

    if r > R0:
        return (Z - 1) / r
    else:
        return (Z - 1) / r * ((3.0 * r / (2.0 * R0)) - 0.5 * (r / R0)**3)


def potential_woods_saxon_neutrons(options, r):

    N = options.num_neutrons
    Z = options.num_protons
    A = N + Z
    R0 = options.ws_r0 * FM_TO_BOHR * A**(1.0/3.0)
    V0_N = options.ws_V0 * MEV_TO_AU * (1.0 - 0.63 * (N - Z) / A)
    xi = options.ws_xi * FM_TO_BOHR
    zeta = 0.263 * (1.0 + 2.0 * (N - Z) / A) * (FM_TO_BOHR * FM_TO_BOHR)

    V_scalar = - V0_N / (1.0 + math.exp((r - R0) / xi))
    V_spin_orbit = (zeta / r) * (V0_N / xi) * math.exp((r - R0) / xi) / (1.0 + math.exp((r - R0) / xi))**2

    return V_scalar, V_spin_orbit


def potential_woods_saxon_protons(options, r):

    N = options.num_neutrons
    Z = options.num_protons
    A = N + Z
    R0 = options.ws_r0 * FM_TO_BOHR * A**(1.0/3.0)
    V0_Z = options.ws_V0 * MEV_TO_AU * (1.0 + 0.63 * (N - Z) / A)
    xi = options.ws_xi * FM_TO_BOHR
    zeta = 0.263 * (1.0 + 2.0 * (N - Z) / A) * (FM_TO_BOHR * FM_TO_BOHR)

    V_scalar = - V0_Z / (1.0 + math.exp((r - R0) / xi))
    V_spin_orbit = (zeta / r) * (V0_Z / xi) * math.exp((r - R0) / xi) / (1.0 + math.exp((r - R0) / xi))**2

    return V_scalar, V_spin_orbit


def numerov_solver(options, N, rmax_fm, mass, L, J, particles):

    rmax = rmax_fm * FM_TO_BOHR
    h = 1.0 / (N + 1)
    gamma2 = 2 * mass * rmax**2

    A = get_tridiagonal_matrix(N, -2.0 / (h * h), 1.0 / (h * h))
    B = get_tridiagonal_matrix(N, 10.0 / 12.0, 1.0 / 12.0)
    V = np.zeros((N, N))

    #
    # construct potential energy matrix
    #
    for i in range(0,N):
        ri = (i + 1) * h * rmax
        E_rot = 1.0 / (2.0 * mass * ri * ri) * L * (L + 1)
        if particles == 'neutron':
            V_sc, V_so = potential_woods_saxon_neutrons(options, ri)
            if J / 2.0 < L:
                V[i, i] = V_sc + E_rot + 0.5 * L * V_so
            else:
                V[i, i] = V_sc + E_rot - 0.5 * (L + 1) * V_so
        else:
            V_sc, V_so = potential_woods_saxon_protons(options, ri)
            V_c = potential_coulomb_protons(options, ri)
            if J / 2.0 < L:
                V[i, i] = V_sc + V_c + E_rot + 0.5 * L * V_so
            else:
                V[i, i] = V_sc + V_c + E_rot - 0.5 * (L + 1) * V_so

    #
    # construct kinetic energy matrix, then hamiltonian
    # H = -1/gamma2 B^-1 A + V
    #
    Binv = numpy.linalg.inv(B)
    H = -1.0 / gamma2 * Binv.dot(A) + V

    #
    # diagonalization
    #
    e, v = numpy.linalg.eigh(H)
    e *= AU_TO_MEV

    return e, v


"""
parses input file.

syntax:
protons <int Z>
neutron <int N>
grid_size <int npoints>
rmax <float rmax>
"""
def read_input_file(path):
    file = open(path, 'r')
    options = Options()

    for line in file.readlines():
        command = line.lower().strip().split('#')[0].split()
        if not command:
            continue

        if command[0] == 'protons':
            options.num_protons = int(command[1])
        elif command[0] == 'neutrons':
            options.num_neutrons = int(command[1])
        elif command[0] == 'grid_size':
            options.grid_size = int(command[1])
        elif command[0] == 'lmax':
            options.lmax = int(command[1])
        elif command[0] == 'rmax':
            options.rmax = float(command[1])
        else:
            raise Exception('unknown keyword: %s' % (command[0]))

    file.close()
    return options


def angular_momentum_string(L):
    return "spdfghiklmnop"[L]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: %s <input-file>' % (sys.argv[0]))
        sys.exit(1)

    print()
    print(' --------------------------------')
    print(' obolochki v 0.1                 ')
    print(' a. oleynichenko, 18 may 2023    ')
    print(' nrc "kurchatov institute" - pnpi')
    print(' --------------------------------')
    print()

    options = read_input_file(sys.argv[1])
    options.print()

    for particle_type in ['proton', 'neutron']:
        roots = []
        all_roots = []
        for L in range(0, options.lmax + 1):
            for J in range(abs(2*L - 1), 2*L + 2, 2):
                e, psi = numerov_solver(options, options.grid_size, options.rmax, options.neutron_mass, L, J, particle_type)
                for n in range(0,len(e)):
                    all_roots.append((e[n], L, J, psi[:, n]))

        all_roots_sorted = sorted(all_roots, key=lambda x: x[0])

        sum_particles = 0
        max_sum = options.num_neutrons if particle_type == 'neutron' else options.num_protons

        npoints = options.grid_size
        density = np.array(np.zeros(npoints))
        rpoints = np.linspace(0, options.rmax, npoints + 2)[1:npoints+1]
        count_L = [0 for l in range(0,options.lmax+1)]

        print('\n %s shells' % particle_type)
        print('           nlj   occ    E(MeV)     <r>')
        for i, root in enumerate(all_roots_sorted):
            E = root[0]
            l = root[1]
            J = root[2]
            psi = root[3]
            if E >= 0.0:
                break

            # principal quantum number
            if J / 2.0 > l:
                count_L[l] += 1
            n = count_L[l]

            # occupation number
            occ = J + 1
            if (sum_particles + occ) > max_sum:
                occ = max_sum - sum_particles
            sum_particles += occ

            # calculate expectation values
            expect_r = 0.0
            for j in range(0, options.grid_size):
                rj = (j + 1) * options.rmax / (options.grid_size + 1)
                expect_r += rj * psi[j]**2

            nlj_label = '%2d%1s_%d/2' % (n, angular_momentum_string(l), J)
            print(' %2d    %-10s %2d  %8.2f%8.2f' % (i + 1, nlj_label, occ, E, expect_r))

            f = open('R_%d%s%s.dat' % (n, angular_momentum_string(l), '+' if J / 2.0 > l else '-'), 'w')
            for j in range(0, options.grid_size):
                rj = (j + 1) * options.rmax / (options.grid_size + 1)
                f.write('%20.12f%20.12f\n' % (rj, psi[j]))
            f.close()

            for j in range(0, options.grid_size):
                rj = (j + 1) * options.rmax / (options.grid_size + 1)
                density[j] += occ * psi[j]**2

        # expectation values
        expect_r2 = 0.0
        expect_1 = 0.0
        for j in range(0, options.grid_size):
            rj = (j + 1) * options.rmax / (options.grid_size + 1)
            expect_r += rj * density[j] / sum_particles
            expect_r2 += rj**2 * density[j] / sum_particles

        print()
        print(' %s <r>   (fm) = %.3f' % (particle_type, expect_r))
        print(' %s rms r (fm) = %.3f' % (particle_type, math.sqrt(expect_r2)))
        print()

        f = open('%s_density.dat' % particle_type, 'w')
        for j in range(0, options.grid_size):
            rj = (j + 1) * options.rmax / (options.grid_size + 1)
            f.write('%20.12f%20.12f\n' % (rj, density[j] / rj**2))
        f.close()

print()
