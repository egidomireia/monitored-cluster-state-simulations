"""Run from the repository root: python scripts/simulate_pauli.py --help"""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from monitored_cluster_states.pauli_measurements import run_simulations


def main():
    parser=argparse.ArgumentParser(description='Fixed physical Pauli sampling with GF(2) entropy.')
    parser.add_argument('--basis',choices=['Y','X','Z'],default='Y')
    parser.add_argument('--sizes',type=int,nargs='+',default=[50,60,70,80,90])
    parser.add_argument('--p-count',type=int,default=100)
    parser.add_argument('--realisations',type=int,default=200)
    parser.add_argument('--seed',type=int,default=20261104)
    parser.add_argument('--periodic',action='store_true')
    parser.add_argument('--checkpoint-every',type=int,default=10)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.p_count<2: parser.error('--p-count must be >=2')
    boundary='pbc' if args.periodic else 'obc'
    root=Path(__file__).resolve().parents[1]
    output=args.output or root/'data_files'/f'pauli_{args.basis.lower()}'/f'Raw_Pauli_{args.basis}_{boundary}.pkl'
    run_simulations(output,sizes=args.sizes,probabilities=np.linspace(0,1,args.p_count),
                    basis=args.basis,periodic=args.periodic,realisations=args.realisations,
                    seed=args.seed,checkpoint_every=args.checkpoint_every)


if __name__=='__main__': main()
