#!/bin/sh
#
#########################################################################
# Copyright (C) 2011-2013:
#
#	Edgardo Mejia-Roa(*), Carlos Garcia, Jose Ignacio Gomez,
#	Manuel Prieto, Francisco Tirado and Alberto Pascual-Montano(**).
#
#	(*)  ArTeCS Group, Complutense University of Madrid (UCM), Spain.
#	(**) Functional Bioinformatics Group, Biocomputing Unit,
#		National Center for Biotechnology-CSIC, Madrid, Spain.
#
#	E-mail for E. Mejia-Roa: <edgardomejia@fis.ucm.es>
#	E-mail for A. Pascual-Montano: <pascual@cnb.csic.es>
#
#
# This file is part of bioNMF-mGPU..
#
# BioNMF-mGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BioNMF-mGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BioNMF-mGPU.  If not, see <http://www.gnu.org/licenses/>.
#
#########################################################################


#############################
# Default paths
#############################

cuda_install_path="/usr/local/cuda-5.5"


#############################
# Changes the default path
# if another is provided.
#############################

if (( $# )) ; then
	cuda_install_path="$1"
fi

if [ ! -d "${cuda_install_path}" ] ; then
	echo "${cuda_install_path}: CUDA directory not found." >&2
	return 1
fi

echo "Setting CUDA install path to: ${cuda_install_path}"



#############################
# Sets the new environment
#############################

export LD_LIBRARY_PATH="${cuda_install_path}"/lib:${LD_LIBRARY_PATH}

export DYLD_LIBRARY_PATH="${cuda_install_path}"/lib:${DYLD_LIBRARY_PATH}
