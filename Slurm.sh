#!/bin/bash

#SBATCH --job-name=coupling_uedge_heat

#SBATCH --partition=serial                    
#SBATCH --time=48:00:00

#SBATCH --output=coupling_uedge_heat_output.log  
#SBATCH --error=coupling_uedge_heat_error.log    

#SBATCH --mail-type=ALL                      
#SBATCH --mail-user=islam9@llnl.gov            

#SBATCH --ntasks=1                            
#SBATCH --cpus-per-task=1                  

#SBATCH --mem=4G                              

#SBATCH --export=ALL                          

export OPENBLAS_NUM_THREADS=1                                        

export LD_LIBRARY_PATH=/usr/local/anaconda/anaconda-3.10/lib:$LD_LIBRARY_PATH

python3 coupling_UEDGE_heat.py              
