#python3 mountain_solver.py --lr 0.0001 --lr_decay 0.99 --lr_decay_step 1000 --eps 0.9 --eps_decay 0.99 --eps_decay_step 10 --n_eps 20000 --save_path ./adam1 --freeze_time 1000 --stime 10
#python3 mountain_solver.py --lr 0.00009 --lr_decay 0.99 --lr_decay_step 1000 --eps 0.0001 --eps_decay 0.99 --eps_decay_step 10 --n_eps 2000 --save_path ./adam2 --load_path ./adam1 --freeze_time 500 --stime 10
#python3 mountain_solver.py --lr 0.000087 --lr_decay 0.99 --lr_decay_step 200 --eps 0.001 --eps_decay 0.99 --eps_decay_step 100 --n_eps 1000 --save_path ./adam3 --load_path ./adam2 --freeze_time 100 --stime 10
#python3 mountain_solver.py --lr 0.000087 --lr_decay 0.99 --lr_decay_step 200 --eps 0.001 --eps_decay 0.99 --eps_decay_step 100 --n_eps 1000 --save_path ./adam4 --load_path ./adam3 --freeze_time 10 --stime 10
#python3 mountain_solver.py --lr 0.3 --lr_decay 0.0001 --eps 0.1 --eps_decay 0  --n_eps 5000 --save_path ./test_lr_1 --freeze_time 100 --stime 10
#python3 mountain_solver.py --lr 0.3 --lr_decay 0.0005 --eps 0.1 --eps_decay 0  --n_eps 5000 --save_path ./test_lr_2 --freeze_time 100 --stime 10
#python3 mountain_solver.py --lr 0.3 --lr_decay 0.001 --eps 0.1 --eps_decay 0  --n_eps 7000 --save_path ./test_lr_3 --freeze_time 100 --stime 10
#python3 mountain_solver.py --lr 0.3 --lr_decay 0.005 --eps 0.1 --eps_decay 0  --n_eps 5000 --save_path ./test_lr_4 --freeze_time 100 --stime 10
#python3 mountain_solver.py --lr 0.3 --lr_decay 0.0001 --eps 0.3 --eps_decay 0.0001  --n_eps 10000 --save_path ./again --freeze_time 1 --stime 10 --gamma 0.98

#python3 mountain_solver.py --lr 0.3 --lr_decay 0.0001 --eps 0.3 --eps_decay 0.0001  --n_eps 10000 --load_path ./tanh_activation --freeze_time 1 --stime 10 --gamma 0.98 --save_path ./visualize
python3 mountain_solver.py --lr 0.00025 --lr_decay 0 --eps 0.0 --eps_decay 0.0001  --n_eps 5000  --freeze_time 1000 --stime 10 --gamma 0.99  --test --save_path ./visualize



