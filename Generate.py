from scipy.interpolate import interp1d
from smt.sampling_methods import LHS
from classy import Class
from glob import glob
from tqdm import tqdm
import numpy as np
import subprocess
import json
import csv
import sys
import os

# As 10 to 30
# OmM 0.2, 0.4
# axm 1e-22, 1e-28
# axf 0.01, 1.0 

class Generate: 
    def __init__(self, 
                 # Simulation parameters
                 label="sim",          
                 outputfolder="output",
                 axioncamb_folder = "/mn/stornext/u3/dennisfr/local/axionCAMB",
                 cola_folder = "/mn/stornext/u3/dennisfr/local/FML/FML/COLASolver",
               
                 # Cosmological parameters 
                 h = 0.6711,
                 Omega_b = 0.049,
                 Omega_cdm = 0.2637,
                 Omega_ncdm = 0.0048,
                 w0 = -1.0,
                 wa = 1.0,
                 A_s = 2.3e-9,
                 n_s = 0.966,
                 N_ncdm = 1,
                 T_cmb = 2.7255, 
                 k_pivot = 0.05,
                 Omega_k = 0.0,
                 f_axion = 0.2,
                 m_axion = 1e-24,
                 # COLA parameters
                 boxsize = 350.0,
                 cola = "true",
                 cosmology = "w0waCDM",
                 gravity = "GR",
                 npart = 640,
                 nmesh = 640,
                 ntimesteps = 30,
                 seed = 1234567,
                 zini = 30.0
                ):
        
        self.label = label
        self.label_LCDM = label + "_LCDM"
        
        self.outputfolder = outputfolder
        if not os.path.isdir(self.outputfolder):
            subprocess.run(["mkdir", self.outputfolder])
            
        # make symbolic link to HighLExtrapTemplate_lenspotentialCls.dat
        # TODO: check if .dat file is there 
        subprocess.run(["ln", "-s", axioncamb_folder + "/HighLExtrapTemplate_lenspotentialCls.dat", "."])
        
        # define exe path
        self.axioncamb_exe = axioncamb_folder + "/camb"
        self.cola_exe = cola_folder + "/nbody"
        
        # Cosmological parameters
        self.h          = h          # Hubble H0 / 100km/s/Mpc
        self.Omega_b    = Omega_b    # Baryon density
        self.Omega_cdm  = Omega_cdm  # CDM density
        self.Omega_ncdm = Omega_ncdm # 0.0048       # Massive neutrino density
        self.Omega_k    = Omega_k    # Curvature density
        self.Omega_fld  = 1.0 - (Omega_b + Omega_cdm + Omega_ncdm + Omega_k)
        
        self.w0      = w0
        self.wa      = wa
        self.A_s     = A_s
        self.n_s     = n_s
        self.k_pivot = k_pivot         # Pivot scale in 1/Mpc
        self.T_cmb   = T_cmb           # CMB temperature
        self.Neff    = 3.046
        self.N_ncdm  = 1
        # N_ur = 3.044
        # N_ncdm = 0
        # Neff remove
        self.N_ur    = self.Neff - self.N_ncdm   # Effective number of MASSLESS neutrino species

        self.f_axion = f_axion
        self.m_axion = m_axion

        # COLA parameters
        self.boxsize    = boxsize
        self.cola       = cola
        self.cosmology  = cosmology
        self.gravity    = gravity
        self.npart      = npart
        self.nmesh      = nmesh
        self.ntimesteps = ntimesteps
        self.seed       = seed
        self.zini       = zini
        
        zarr = np.exp(-np.linspace(np.log(1.0/(1.0+zini)),np.log(1.0),100))-1.0
        for i in range(len(zarr)):
            zarr[i] = round(zarr[i],3)
        self.zarr = zarr
        
    
    def make_inifiles(self,
         label,
         LCDM=False):

        if LCDM:
          m_axion = 1e-20
          f_axion = 0.001
        else:
          m_axion = self.m_axion
          f_axion = self.f_axion
        
        root = self.outputfolder + "/axioncamb_"+label
        axioncambinput = "\
# Label                                                                  \n\
output_root                   = "+root+"                                 \n\
                                                                         \n\
# Axion parameters                                                       \n\
m_ax                          = "+str(m_axion)+"                         \n\
axfrac                        = "+str(f_axion)+"                         \n\
                                                                         \n\
# Cosmology parameters                                                   \n\
omch2                         = "+str(0.0)+"                             \n\
omdah2                        = "+str(self.Omega_cdm*self.h*self.h)+"    \n\
ombh2                         = "+str(self.Omega_b*self.h*self.h)+"      \n\
omnuh2                        = "+str(self.Omega_ncdm*self.h*self.h)+"   \n\
hubble                        = "+str(100*self.h)+"                      \n\
massive_neutrinos             = "+str(self.N_ncdm)+"                     \n\
massless_neutrinos            = "+str(self.N_ur)+"                       \n\
massive_nu_approx             = 1                                        \n\
w                             = "+str(self.w0)+"                         \n\
wa                            = "+str(self.wa)+"                         \n\
                                                                         \n\
# Normalization                                                          \n\
pivot_scalar                  = "+str(self.k_pivot)+"                    \n\
scalar_amp(1)                 = "+str(self.A_s)+"                        \n\
scalar_spectral_index(1)      = "+str(self.n_s)+"                        \n\
                                                                         \n\
# Non-linear?                                                            \n\
do_nonlinear                  = 0                                        \n\
halofit_version               = 1                                        \n\
                                                                         \n\
# Parameters we do not have to touch                                     \n\
omk                           = 0                                        \n\
get_scalar_cls                = T                                        \n\
get_vector_cls                = F                                        \n\
get_tensor_cls                = F                                        \n\
get_transfer                  = T                                        \n\
do_lensing                    = F                                        \n\
l_max_scalar                  = 2500                                     \n\
l_max_tensor                  = 1500                                     \n\
k_eta_max_tensor              = 3000                                     \n\
use_physical                  = T                                        \n\
cs2_lam                       = 1                                        \n\
use_axfrac                    = T                                        \n\
axion_isocurvature            = F                                        \n\
alpha_ax                      = 0                                        \n\
Hinf                          = 13.7                                     \n\
omaxh2                        = 0.0                                      \n\
temp_cmb                      = 2.725                                    \n\
helium_fraction               = 0.24                                     \n\
share_delta_neff              = T                                        \n\
nu_mass_eigenstates           = 1                                        \n\
nu_mass_fractions             = 1                                        \n\
initial_power_num             = 1                                        \n\
pivot_tensor                  = 0.05                                     \n\
scalar_nrun(1)                = 0                                        \n\
tensor_spectral_index(1)      = 0                                        \n\
initial_ratio(1)              = 0                                        \n\
tens_ratio                    = 0                                        \n\
reionization                  = T                                        \n\
re_use_optical_depth          = T                                        \n\
re_optical_depth              = 0.078                                    \n\
re_redshift                   = 11                                       \n\
re_delta_redshift             = 1.5                                      \n\
re_ionization_frac            = -1                                       \n\
RECFAST_fudge                 = 1.14                                     \n\
RECFAST_fudge_He              = 0.86                                     \n\
RECFAST_Heswitch              = 6                                        \n\
RECFAST_Hswitch               = T                                        \n\
initial_condition             = 1                                        \n\
initial_vector                = 1 0 0 0 0 1                              \n\
vector_mode                   = 0                                        \n\
COBE_normalize                = F                                        \n\
CMB_outputscale               = 7.4311e12                                \n\
transfer_high_precision       = F                                        \n\
transfer_kmax                 = 20.0                                     \n\
transfer_k_per_logint         = 10                                       \n\
transfer_interp_matterpower   = T                                        \n\
scalar_output_file            = scalCls.dat                              \n\
vector_output_file            = vecCls.dat                               \n\
tensor_output_file            = tensCls.dat                              \n\
total_output_file             = totCls.dat                               \n\
lensed_output_file            = lensedCls.dat                            \n\
lensed_total_output_file      = lensedtotCls.dat                         \n\
lens_potential_output_file    = lenspotentialCls.dat                     \n\
FITS_filename                 = scalCls.fits                             \n\
do_lensing_bispectrum         = F                                        \n\
do_primordial_bispectrum      = F                                        \n\
bispectrum_nfields            = 2                                        \n\
bispectrum_slice_base_L       = 0                                        \n\
bispectrum_ndelta             = 3                                        \n\
bispectrum_delta(1)           = 0                                        \n\
bispectrum_delta(2)           = 2                                        \n\
bispectrum_delta(3)           = 4                                        \n\
bispectrum_do_fisher          = F                                        \n\
bispectrum_fisher_noise       = 0                                        \n\
bispectrum_fisher_noise_pol   = 0                                        \n\
bispectrum_fisher_fwhm_arcmin = 7                                        \n\
bispectrum_full_output_file   =                                          \n\
bispectrum_full_output_sparse = F                                        \n\
feedback_level                = 1                                        \n\
lensing_method                = 1                                        \n\
accurate_BB                   = F                                        \n\
accurate_polarization         = T                                        \n\
accurate_reionization         = T                                        \n\
do_tensor_neutrinos           = T                                        \n\
do_late_rad_truncation        = T                                        \n\
number_of_threads             = 0                                        \n\
high_accuracy_default         = F                                        \n\
accuracy_boost                = 1                                        \n\
l_accuracy_boost              = 1                                        \n\
l_sample_boost                = 1                                        \n\
"
  
        # Add the redshift output info
        extra=""
        extra += ""
        extra += "transfer_num_redshifts = "+str(len(self.zarr))+" \n"
        for i in range(len(self.zarr)):
            extra += "\
transfer_redshift("+str(i+1)+")          = "+str(self.zarr[i])+"                  \n\
transfer_filename("+str(i+1)+")          = transfer_z"+str(self.zarr[i])+".dat    \n\
transfer_matterpower("+str(i+1)+")       = matterpower_z"+str(self.zarr[i])+".dat \n"
 
        # Save file
        axioncambinput += extra
        axioncamb_paramfile = self.outputfolder + "/axioncamb_"+label+"_paramfile.inp"
        with open(axioncamb_paramfile, 'w') as f:
            f.write(axioncambinput)

        
        # [3] Run axioncamb
        subprocess.run([self.axioncamb_exe, axioncamb_paramfile])
        
        # [4] Make the transferinfofile for COLA
        
        colatransferinfofile = self.outputfolder + " " + str(len(self.zarr))
        zarr = np.flip(self.zarr)
        for _z in zarr:
            filename = "axioncamb_"+label+"_transfer_z"+str(_z)+".dat"
            colatransferinfofile += "\n" + filename + " " + str(_z) 
  
        # [5] Write transferinfo-file needed by COLA (YOU DON'T NEED THIS)
        self.colatransferinfofilename = self.outputfolder + "/axioncamb_transferinfo_"+label+".txt"
        with open(self.colatransferinfofilename, 'w') as f:
            f.write(colatransferinfofile)
            
        # [6] Write the COLA parameterfile
        colafile = "\
------------------------------------------------------------ \n\
-- Simulation parameter file                                 \n\
-- Include other paramfile into this: dofile(\"param.lua\")  \n\
------------------------------------------------------------ \n\
                                                             \n\
-- Don't allow any parameters to take optional values?       \n\
all_parameters_must_be_in_file = true                        \n\
------------------------------------------------------------ \n\
-- Simulation options                                        \n\
------------------------------------------------------------ \n\
-- Label                                                     \n\
simulation_name = \""+label+"\"                         \n\
-- Boxsize of simulation in Mpc/h                            \n\
simulation_boxsize = "+str(self.boxsize)+"                   \n\
                                                             \n\
------------------------------------------------------------ \n\
-- COLA                                                      \n\
------------------------------------------------------------ \n\
-- Use the COLA method                                       \n\
simulation_use_cola = true                                   \n\
simulation_use_scaledependent_cola = "+self.cola+"           \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Choose the cosmology                                      \n\
------------------------------------------------------------ \n\
-- Cosmology: LCDM, w0waCDM, DGP, JBD, ...                   \n\
cosmology_model = \""+self.cosmology+"\"                     \n\
cosmology_OmegaCDM = "+str(self.Omega_cdm)+"                 \n\
cosmology_Omegab = "+str(self.Omega_b)+"                     \n\
cosmology_OmegaMNu = "+str(self.Omega_ncdm)+"                \n\
cosmology_OmegaLambda = "+str(self.Omega_fld)+"              \n\
cosmology_Neffective = "+str(self.Neff)+"                    \n\
cosmology_TCMB_kelvin = "+str(self.T_cmb)+"                  \n\
cosmology_h = "+str(self.h)+"                                \n\
cosmology_As = "+str(self.A_s)+"                             \n\
cosmology_ns = "+str(self.n_s)+"                             \n\
cosmology_kpivot_mpc = "+str(self.k_pivot)+"                 \n\
-- The w0wa parametrization                                  \n\
if cosmology_model == \"w0waCDM\" then                       \n\
cosmology_w0 = "+str(self.w0)+"                              \n\
cosmology_wa = "+str(self.w0)+"                              \n\
end                                                          \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Choose the gravity model                                  \n\
------------------------------------------------------------ \n\
-- Gravity model: GR, DGP, f(R), JBD, Geff, ...              \n\
gravity_model = \""+self.gravity+"\"                         \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Particles                                                 \n\
------------------------------------------------------------ \n\
-- Number of CDM+b particles per dimension                   \n\
particle_Npart_1D = "+str(self.npart)+"                      \n\
-- Factor of how many more particles to allocate space       \n\
particle_allocation_factor = 1.25                            \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Output                                                    \n\
------------------------------------------------------------ \n\
-- List of output redshifts                                  \n\
output_redshifts = {0.0}                                    \n\
-- Output particles?                                         \n\
output_particles = false                                      \n\
-- Fileformat: GADGET, FML                                   \n\
output_fileformat = \"GADGET\"                               \n\
-- Output folder                                             \n\
output_folder = \""+self.outputfolder+"\"                    \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Time-stepping                                             \n\
------------------------------------------------------------ \n\
-- Number of steps between the outputs (in output_redshifts) \n\
timestep_nsteps = {"+str(self.ntimesteps)+"}                 \n\
-- The time-stepping method: Quinn, Tassev                   \n\
timestep_method = \"Quinn\"                                  \n\
-- For Tassev: the nLPT parameter                            \n\
timestep_cola_nLPT = -2.5                                    \n\
-- The time-stepping algorithm: KDK                          \n\
timestep_algorithm = \"KDK\"                                 \n\
-- Spacing of the time-steps in 'a': linear, logarithmic, .. \n\
timestep_scalefactor_spacing = \"linear\"                    \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Initial conditions                                        \n\
------------------------------------------------------------ \n\
-- The random seed                                           \n\
ic_random_seed = "+str(self.seed)+"                          \n\
-- The random generator (GSL or MT19937).                    \n\
ic_random_generator = \"GSL\"                                \n\
-- Fix amplitude when generating the gaussian random field   \n\
ic_fix_amplitude = true                                      \n\
-- Mirror the phases (for amplitude-fixed simulations)       \n\
ic_reverse_phases = false                                    \n\
ic_random_field_type = \"gaussian\"                          \n\
-- The grid-size used to generate the IC                     \n\
ic_nmesh = particle_Npart_1D                                 \n\
-- For MG: input LCDM P(k) and use GR to scale back and      \n\
-- ensure same IC as for LCDM                                \n\
ic_use_gravity_model_GR = false                              \n\
-- The LPT order to use for the IC                           \n\
ic_LPT_order = 2                                             \n\
-- The type of input:                                        \n\
-- powerspectrum    ([k (h/Mph) , P(k) (Mpc/h)^3)])          \n\
-- transferfunction ([k (h/Mph) , T(k)  Mpc^2)]              \n\
-- transferinfofile (a bunch of T(k,z) files from CAMB)      \n\
ic_type_of_input = \"transferinfofile\"                      \n\
ic_type_of_input_fileformat = \"AXIONCAMB\"                  \n\
-- Path to the input                                         \n\
ic_input_filename = \""+self.colatransferinfofilename+"\"    \n\
-- The redshift of the P(k), T(k) we give as input           \n\
ic_input_redshift = 0.0                                      \n\
-- The initial redshift of the simulation                    \n\
ic_initial_redshift = "+str(self.zini)+"                     \n\
-- Normalize wrt sigma8?                                     \n\
-- If ic_use_gravity_model_GR then this is the sigma8 value  \n\
-- in a corresponding GR universe!                           \n\
ic_sigma8_normalization = false                              \n\
ic_sigma8_redshift = 0.0                                     \n\
ic_sigma8 = 0.83                                             \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Force calculation                                         \n\
------------------------------------------------------------ \n\
-- Grid to use for computing PM forces                       \n\
force_nmesh = "+str(self.nmesh)+"                            \n\
-- Density assignment method: NGP, CIC, TSC, PCS, PQS        \n\
force_density_assignment_method = \"CIC\"                    \n\
-- The kernel to use when solving the Poisson equation       \n\
force_kernel = \"continuous_greens_function\"                \n\
-- Include the effects of massive neutrinos when computing   \n\
-- the density field (density of mnu is linear prediction    \n\
-- Requires: transferinfofile above (we need all T(k,z))     \n\
force_linear_massive_neutrinos = true                        \n\
                                                             \n\
------------------------------------------------------------ \n\
-- On the fly analysis                                       \n\
------------------------------------------------------------ \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Halofinding                                               \n\
------------------------------------------------------------ \n\
fof = false                                                  \n\
fof_nmin_per_halo = 20                                       \n\
fof_linking_length = 0.2                                     \n\
fof_nmesh_max = 0                                            \n\
fof_buffer_length_mpch = 3.0                                 \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Power-spectrum evaluation                                 \n\
------------------------------------------------------------ \n\
pofk = false                                                 \n\
pofk_nmesh = force_nmesh                                     \n\
pofk_interlacing = true                                      \n\
pofk_subtract_shotnoise = true                               \n\
pofk_density_assignment_method = \"PCS\"                     \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Power-spectrum multipole evaluation                       \n\
------------------------------------------------------------ \n\
pofk_multipole = false                                       \n\
pofk_multipole_nmesh = force_nmesh                           \n\
pofk_multipole_interlacing = true                            \n\
pofk_multipole_subtract_shotnoise = false                    \n\
pofk_multipole_ellmax = 4                                    \n\
pofk_multipole_density_assignment_method = \"PCS\"           \n\
                                                             \n\
------------------------------------------------------------ \n\
-- Bispectrum evaluation                                     \n\
------------------------------------------------------------ \n\
bispectrum = false                                           \n\
bispectrum_nmesh = 128                                       \n\
bispectrum_nbins = 10                                        \n\
bispectrum_interlacing = true                                \n\
bispectrum_subtract_shotnoise = false                        \n\
bispectrum_density_assignment_method = \"PCS\"               \n\
  """
        self.colainputfile = self.outputfolder + "/cola_input_"+label+".lua"
        with open(self.colainputfile, 'w') as f:
            f.write(colafile)
            
            
            
    def run_sim(self):
        mpirunexe="/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/bin/mpirun"
        colaexe="/mn/stornext/u3/dennisfr/local/FML/FML/COLASolver/nbody"

        self.make_inifiles(self.label)
        command = "OMP_NUM_THREADS=4 {} -np 32 {} {}".format(mpirunexe, self.cola_exe, self.colainputfile)
        os.system(command)
        
        self.make_inifiles(self.label_LCDM, LCDM=True)
        command = "OMP_NUM_THREADS=4 {} -np 32 {} {}".format(mpirunexe, self.cola_exe, self.colainputfile)
        os.system(command)
        

    def make_LHS(self,
          OmegaM_limits = [0.2, 0.4],
          logAs_limits     = [np.log10(10e-10), np.log10(30e-10)],
          fax_limits    = [0.001, 1.0],
          logmax_limits    = [-28, -22],
          Nsamples=500,
          prefix="Geff"):
        """
        make_LHS: Makes a latin hypercube of samples given by params, with Nsamples. File is saved with prefix.
        ARGS: 
            - Nsamples (int) : number of samples 
            - prefix (str)   : prefix to save LHS to 
        """
        dictfile = "LHS_"+prefix+"_"+str(Nsamples)+".json"
        params = {'Omega_M': OmegaM_limits,
                 'logA_s': logAs_limits,
                 'f_ax': fax_limits,
                 'logm_ax': logmax_limits}
        ranges = []
        [ranges.append(params[key]) for key in params]
        sampling = LHS(xlimits=np.array(ranges))
        all_samples = sampling(Nsamples)
        
        # save all_samples
        parameters_for_all_samples = {}
        num = 1
        for sample in all_samples:
            name = "sim"+str(num)
            
            dict = {}
            dict['Omega_M'] = sample[0]
            dict['A_s']     = 10**sample[1]
            dict['f_ax']    = sample[2]
            dict['m_ax']    = 10**sample[3]
            
            parameters_for_all_samples[name] = dict
            num += 1
        
        # save file
        data = json.dumps(parameters_for_all_samples)
        with open("parameters_of_samples.json","w") as f:
          f.write(data)

        return all_samples

    def run_pipeline(self):
        mpirunexe="/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/bin/mpirun"
        colaexe="/mn/stornext/u3/dennisfr/local/FML/FML/COLASolver/nbody"

        # save params so we can reset after run_pipeline
        _label = self.label
        _OmM = self.Omega_cdm
        _As  = self.A_s
        _fax = self.f_axion
        _max = self.m_axion
        _Omncdm = self.Omega_ncdm
        
        self.Omega_ncdm = 0.0 # set to zero to speed up sims
        
        with open("parameters_of_samples.json") as infile:
            all_samples = json.load(infile)
        
        for key in all_samples:
            if glob("output_main/pofk_"+key+"_cb_z0.000.txt"):
                 continue

            self.label = key
            self.label_LCDM = key + "_LCDM"
            
            self.Omega_cdm = all_samples[key]['Omega_M']
            self.A_s       = all_samples[key]['A_s']
            self.f_axion   = all_samples[key]['f_ax']
            self.m_axion   = all_samples[key]['m_ax']
            
            # run LCDM and mixed cosmo sims
            self.make_inifiles(self.label)
            command = "OMP_NUM_THREADS=4 {} -np 32 {} {}".format(mpirunexe, self.cola_exe, self.colainputfile)
            os.system(command)
        
            #self.make_inifiles(self.label_LCDM, LCDM=True)
            command = "OMP_NUM_THREADS=4 {} -np 32 {} {}".format(mpirunexe, self.cola_exe, self.colainputfile)
            os.system(command)
            
            
        # reset values
        self.label = _label
        self.label_LCDM = _label+"_LCDM"
        self.Omega_cdm = _OmM
        self.A_s = _As
        self.f_axion = _fax
        self.m_axion = _max
        self.Omega_ncdm = _Omncdm
        
    def make_test_train(self):
        np.random.seed(0)
        i_tst = [np.random.randint(0, 500*15) for dummy in range(30)]
        i_val = i_tst[:15]
        i_tst = i_tst[15:]
        
        data = []
        data_tst = []
        data_val = []
        
        head = ['OmegaCDM', 'logA_s', 'f_ax', 'logm_ax', 'z', 'k', 'pfrac']

        z = ["0.000", "0.069", "0.148", "0.240", "0.348", 
             "0.476", "0.632", "0.824", "1.067", "1.385",
             "1.818", "2.444", "3.429", "5.200", "6.750"]
    
        z_camb = ["0.0", "0.072", "0.149", "0.231", "0.32",
                 "0.465", "0.625", "0.803", "1.072", "1.38",
                 "1.831", "2.486", "3.444", "5.286", "6.741"]
        
        k_lin = np.linspace(np.log10(1e-3), np.log10(0.02), 128)
        
        count = 0
        
        with open("parameters_of_samples.json") as infile:
            params = json.load(infile)
        
        i = 0
        for key in tqdm(params):
            p = params[key]
            
            Omega_CDM = p['Omega_M']
            A_s       = p['A_s']
            f_ax      = p['f_ax']
            m_ax      = p['m_ax']

            for zi, zi_camb in zip(z, z_camb):
                try: 
                    file_cola = self.outputfolder+"/pofk_"+key+"_cb_z"+zi+".txt"
                    data_cola = np.loadtxt(file_cola)
                    file_cola_LCDM = self.outputfolder+"/pofk_"+key+"_LCDM_cb_z"+zi+".txt"
                    data_cola_LCDM = np.loadtxt(file_cola_LCDM)

                    file_camb = self.outputfolder+"/axioncamb_"+key+"_matterpower_z"+zi_camb+".dat"
                    data_camb = np.loadtxt(file_camb)
                    file_camb_LCDM = self.outputfolder+"/axioncamb_"+key+"_LCDM_matterpower_z"+zi_camb+".dat"
                    data_camb_LCDM = np.loadtxt(file_camb_LCDM)
                except: 
                    count += 1
                    print("Skipping " + key + " at redshift " + zi)
                    continue
                
                k = np.linspace(np.log10(data_cola[0,0]), np.log10(data_cola[-1,0]), 128)
                
                r_cola = interp1d(np.log10(data_cola[:,0]), data_cola[:,1])(k) / interp1d(np.log10(data_cola_LCDM[:,0]), data_cola_LCDM[:,1])(k)
                r_camb = interp1d(np.log10(data_camb[:,0]), data_camb[:,1])(k_lin) / interp1d(np.log10(data_camb_LCDM[:,0]), data_camb_LCDM[:,1])(k_lin)
                    
                # Append data routine
                for ki, ri in zip(k_lin, r_camb):
                    data_list = [Omega_CDM, np.log10(A_s), f_ax, np.log10(m_ax), zi,  10**ki, ri]
                    if i in i_tst:
                        data_tst.append(data_list)
                    elif i in i_val:
                        data_val.append(data_list)
                    else:
                        data.append(data_list)
                        
                for ki, ri in zip(k, r_cola):
                    data_list = [Omega_CDM, np.log10(A_s), f_ax, np.log10(m_ax), zi, 10**ki, ri]
                    if i in i_tst:
                        data_tst.append(data_list)
                    elif i in i_val:
                        data_val.append(data_list)
                    else:
                        data.append(data_list)
                i += 1
                                    
        # WRITE TO CSV
        print("Writing test data to file")
        with open("data/test.csv", "x", encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(head)
            writer.writerows(data_tst)
            
        print("Writing train data")
        with open("data/train.csv", 'x', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(head)        
            writer.writerows(data)    

        print("Writing validation data")
        with open("data/val.csv", 'x', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(head)          
            writer.writerows(data_val)   
            
        print(count, " files could not be used")
                    
                




