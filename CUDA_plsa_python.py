# -*- coding: utf-8 -*-
# Importing PLSA
import numpy as np
import argparse
import os
import sys
sys.path.append("./auxil")
from plsa_cuda import dpLSA, pLSA, initGPU
from aux import load_dataset, generate_inputs, get_metrics, set_seed

# Args dictionaries

modes = ('pLSA', 'dpLSA')

inputs =	("samson",
  "jasper",
  "urban",
  "cuprite")

########################################################################################
##########################     Argparser      #########################################
########################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("mode", help="pLSA | dpLSA")
parser.add_argument("image", help="Image to use")
parser.add_argument("id", help="Execution identifier")
parser.add_argument("--show_images", dest = 'show_images', default=False, action='store_true')
parser.add_argument("--reg_pwz_level1",  default=0.0, type=float)
parser.add_argument("--reg_pzd_level1",  default=0.0, type=float)
parser.add_argument("--reg_pwz_level2",  default=0.0, type=float)
parser.add_argument("--reg_pzd_level2",  default=0.0, type=float)
parser.add_argument("--pca_comp", help="Bands to reduce with Principal Component Analysis (PCA)", default=None, type=int)
parser.add_argument("--dplsa_dim", help="Dimension to go on first step", default=1000, type=int)
parser.add_argument("--seed", help="Random initializer seed", default=0, type=int)

args = parser.parse_args()

########################################################################################
##########################     PARAMETROS      #########################################
########################################################################################
maxIteration1 = 100
maxIteration2 = 100
r1d = args.reg_pwz_level1 # regularizador1 dpLSA
r2d = args.reg_pzd_level1 # regularizador2 dpLSA
r1 = args.reg_pwz_level2 # regularizador1 pLSA
r2 = args.reg_pzd_level2 # regularizador2 pLSA
set_seed(args.seed) # Initializing numpy random seed

if (not (args.image in inputs)):
    print ("Invalid image, please use one of the above")
    print (  inputs)
    exit()

if (not (args.mode in modes)):
    print ("Only pLSA or dpLSA are valid modes, returning")
    exit()

del modes, inputs

########################################################################################
##########################     Inicializacion      #########################################
########################################################################################

(image, gt, K_img) = load_dataset(args.image)
shape_image = image.shape

output_folder = os.path.join('outputs/', (args.mode + '/'))
if not (os.path.exists(output_folder)):
    os.makedirs(output_folder)
if not (os.path.exists('outputs/images/')):
    os.makedirs('outputs/images/')

output_abundances_file_name = 'abundances_' + args.image + '_' + args.id + '.npz'
output_abundances_file = os.path.join(output_folder, output_abundances_file_name)
output_endmembers_file_name = 'endmembers_' + args.image + '_' + args.id + '.npz'
output_endmembers_file = os.path.join(output_folder, output_endmembers_file_name)

if (args.mode == 'pLSA'):
    K = K_img
elif (args.mode == 'dpLSA'):
    K = args.dplsa_dim

if args.pca_comp != None:
	from sklearn.decomposition import PCA
	image = PCA(n_components=args.pca_comp).fit_transform(image.reshape(-1, shape_image[-1])).reshape(shape_image[0], shape_image[1], args.pca_comp)
	shape_image = image.shape


#print ('Running {} over dataset {}'.format(args.mode, args.image))

(p, X, denominators, theta, lamda, N, M) = generate_inputs(image, shape_image, K)
initGPU()

(abundances, endmembers, time) = pLSA(lamda, theta, p, X, denominators, shape_image, N, M, K, maxIteration2, r1, r2)
print("CUDA VERSION: Epochs", maxIteration2, "Time per epoch", np.round(time/maxIteration2, 2), "Total time", np.round(time, 2))



