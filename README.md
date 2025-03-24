# [Locally Convex Global Loss Network for Decision-Focused Learning](https://arxiv.org/abs/2403.01875)


This repository contains the source code for "LCGLN for DFL", accepted in AAAI-25 for Oral Presentation.

The structure of our repository is as follows:
```
├── lcgln/                        # Files for learning loss
│   ├── Networks.py         
│   ├── dataloader.py        
│   └── losses.py
│
├── problems/                     # Optimization problem files
│   ├── BudgetAllocation.py        
│   ├── Newsvendor.py             # Inventory Stock Problem
│   ├── PThenO.py                 # Abstract method class for DFL problem formulation
│   └── PortfolioOpt.py
│   
├── utils/                  
│   ├── SubmodularOptimizer.py    # Differentiable solver for Budget Allocation problem
│   └── utils.py
│
├── README.md
└── main.py
```

We used the following version of python and pytorch:
```
python=3.8
pytorch=2.3
```

As an example, you can run the portfolio optimization in our paper as:
```
python main.py --problem "portfolio" --instances 400 --testinstances 400 --valfrac 0.5 --numsamples 32 --samplinglr 1 --intermediatesize 500
```

If you find our paper or this code helpful, kindly consider citing our paper:
```
@misc{jeon2025locallyconvexgloballoss,
      title={Locally Convex Global Loss Network for Decision-Focused Learning}, 
      author={Haeun Jeon and Hyunglip Bae and Minsu Park and Chanyeong Kim and Woo Chang Kim},
      year={2025},
      eprint={2403.01875},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.01875}, 
}
```

Thank you.
