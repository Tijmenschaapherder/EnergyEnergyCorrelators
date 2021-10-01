

import eec
import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

# load tree 
# fill path to tree
file = uproot.open("~/JetToyHI/Tree.root:jetTree;1")

# load branches 
particle_pt = file["sigJetConstPt"].array()
rapidity = file["sigJetConstRap"].array()
azimuthal_angle = file["sigJetConstPhi"].array()
sigJetPt = file["sigJetPt"].array()
sigJetPDG = file["sigJetConstPDG"].array()
sigJetEta = file["sigJetEta"].array()


# Find longest list in list
def find_max_particles(list):
    list_len = [len(i) for i in list]
    return max(list_len)
  
# pt cuts in GeV
minjetpt = 100 
maxjetpt = 120
minparticlept = 1
minjeteta = -2 
maxjeteta = 2

# flatten data
flatpt = ak.flatten(particle_pt)
flatrap = ak.flatten(rapidity)
flatphi = ak.flatten(azimuthal_angle)
flatjetpt = ak.flatten(sigJetPt)
flatPDG = ak.flatten(sigJetPDG)
flatEta = ak.flatten(sigJetEta)
   
# Define highest number of particles in event and number of events
max_particles = find_max_particles(flatpt)
number_jets = len(flatjetpt)


# Make list of zeroes of appropriate size
jets = np.zeros((number_jets,max_particles,4))

# rearrange data to fit eec function   
for jet in range(0,len(flatpt)):
    for particle in range(0, len(flatpt[jet])):
      if flatpt[jet, particle] > minparticlept:
          jets[jet, particle,0] = flatpt[jet][particle] 
          jets[jet, particle,1] = flatrap[jet][particle]
          jets[jet, particle,2] = flatphi[jet][particle]
          jets[jet, particle,3] = flatPDG[jet][particle]
      else: 
          jets[jet, particle,0] = 0
          jets[jet, particle,1] = 0
          jets[jet, particle,2] = 0
          jets[jet, particle,3] = 0
            
# selection on jets     
selectedjets = jets[ak.where((flatjetpt > minjetpt) & (flatjetpt < maxjetpt) & (flatEta > minjeteta) & (flatEta < maxjeteta))]

# plot lay-out        
colors = {2: 'tab:blue', 3: 'tab:green', 4: 'tab:red'}
errorbar_opts = {
   'fmt': 'o',
   'lw': 1.5,
   'capsize': 1.5,
   'capthick': 1,
   'markersize': 1.5,
}

# EECLongestSide instance
eec_ls = eec.EECLongestSide(2, 100, axis_range=(1e-5, 1))

# multicore compute
eec_ls(selectedjets)

# scale eec for plot 
eec_ls.scale(1/eec_ls.sum())

# gets bins
midbins, bins = eec_ls.bin_centers(), eec_ls.bin_edges()
binwidths = np.log(bins[1:]) - np.log(bins[:-1])

# get EEC hist and (rough) error estimate
# argument 0 means get histogram 0 (there can be multiple if using asymmetric vertex powers)
# argument False mean ignore the overflow bins
hist, errs = eec_ls.get_hist_errs(0, False)

# error options 
plt.errorbar(midbins, hist/binwidths,
             xerr=(midbins - bins[:-1], bins[1:] - midbins),
             yerr=errs/binwidths,
             color=colors[2],
             label='N = {}'.format(2),
             **errorbar_opts)

# plot options
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-5, 1)
plt.ylim(1e-7, 1)
plt.xlabel("$\Delta R$")
plt.ylabel('(Normalized) Cross Section')
plt.legend(loc='lower center', frameon=False)
plt.title('2-particle EEC (jetpt: 100 - 120, particle_pt > 1, jet_eta: ± 2)')

plt.show()
