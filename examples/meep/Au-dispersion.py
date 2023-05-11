# Material dispersion example, from the Meep tutorial.  Here, we simply
# simulate homogenous space filled with a dispersive material, and compute
# its modes as a function of wavevector k.  Since omega/c = k/n, we can
# extract the dielectric function epsilon(omega) = (ck/omega)^2.
import meep as mp

cell = mp.Vector3()
resolution = 20

# We'll use a dispersive material with two polarization terms, just for
# illustration.  The first one is a strong resonance at omega=1.1,
# which leads to a polaritonic gap in the dispersion relation.  The second
# one is a weak resonance at omega=0.5, whose main effect is to add a
# small absorption loss around that frequency.


eb = 5.0
wpl = 9.1
gamma = 1.8
susceptibilities = [
    mp.LorentzianSusceptibility(frequency=wpl, gamma=gamma, sigma=-1.0),
]

default_material = mp.Medium(epsilon=eb, E_susceptibilities=susceptibilities)

#eps = eb - wp^2 / (w^2+i*beta*omega) + i sigma/epsilon0/omega

fcen = 1.0
df = 2.0

sources = [
    mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=mp.Vector3())
]

kmin = 0.3
kmax = 2.2
k_interp = 9#9

kpts = mp.interpolate(k_interp, [mp.Vector3(kmin), mp.Vector3(kmax)])

import sys
print('\nkpts=')
for k in kpts:
    print(k)

sim = mp.Simulation(
    cell_size=cell,
    geometry=[],
    sources=sources,
    default_material=default_material,
    resolution=resolution,
)

all_freqs = sim.run_k_points(200, kpts)  # a list of lists of frequencies

print('\n===dispersion relation=====')
for k in range(k_interp):
    nfreq = len(all_freqs[k])
    if nfreq == 1:
        print(f"{kpts[k].x:9.3g}  {all_freqs[k][0].real:12.6g}" )
    else:
        print(f"{kpts[k].x:9.3g}  {all_freqs[k][0].real:12.6g}  {all_freqs[k][1].real:12.6g} " )

sys.exit()

for fs, kx in zip(all_freqs, [v.x for v in kpts]):
    for f in fs:
        print(f"eps:, {f.real:.6g}, {f.imag:.6g}, {(kx / f) ** 2:.6g}")
