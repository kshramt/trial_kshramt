```
$ time python dqn_mountain_car.py --torch-seed=46 --replay-memory-seed=42 --agent-seed=44 --env-seed=43 --alpha=0.01 --epsilon=0.01 --gamma=1 --log-stderr-level=debug --lr=1e-3 --n-batch=30 --n-episodes=3000 --n-log-steps=100 --n-middle=100  --n-replay-memory=20000 --n-target-update-episodes=20 --n-train-steps=1 --dat-file="tmp.dat" --q-target-mode=mnih2015 --dqn-mode=doubledqn
```

```
$ python dqn_maze.py \
   --agent-seed=44 \
   --alpha=0.001 \
   --dqn-mode=doubledqn \
   --env-seed=43 \
   --epsilon=0.5 \
   --gamma=0.99 \
   --log-stderr-level=debug \
   --lr=1e-4 \
   --n-batch=300 \
   --n-episodes=10000 \
   --n-log-steps=30 \
   --n-middle=500 \
   --n-replay-memory=600 \
   --n-steps=30 \
   --n-target-update-episodes=20 \
   --q-target-mode=mnih2015 \
   --replay-memory-seed=42 \
   --torch-seed=46
```

```
$ julia q_maze.jl out.jld2 42 | gnuplot -persist -e 'plot "< cat" w l'
$ julia q_maze_plot.jl out.jld2
      0.00           0.00           0.00           0.00           0.00           0.00           0.00
  0.00   -0.43  -1.00   -0.40  -0.43   -1.00  -0.40   -0.10  -1.00   -0.05  -0.10   -1.00  -0.05    0.00
     -0.43           0.60           0.63          -0.10           0.95           1.00          -0.05

     -1.00          -0.43          -0.40          -1.00          -0.10           0.00          -1.00
  0.00    0.60  -0.43    0.63   0.60   -0.10   0.63    0.95  -0.10    1.00   0.00    0.00   1.00    0.00
     -0.40           0.63           0.66          -0.14           0.90           0.00          -0.10

     -0.43           0.60           0.63          -0.10           0.95           1.00          -0.05
  0.00    0.63  -0.40    0.66   0.63   -0.14   0.66    0.90  -0.14    0.95   0.90   -0.10   0.95    0.00
     -0.37           0.66           0.70          -0.19           0.86           0.90          -0.14

     -0.40           0.63           0.66          -0.14           0.90           0.95          -0.10
  0.00    0.66  -0.37    0.70   0.66   -0.19   0.70    0.86  -0.19    0.90   0.86   -0.14   0.90    0.00
     -0.34           0.70           0.74           0.77           0.81           0.86          -0.19

     -0.37           0.66           0.70          -0.19           0.86           0.90          -0.14
  0.00    0.70  -0.34    0.74   0.70    0.77   0.74    0.81   0.77    0.86   0.81   -0.19   0.86    0.00
     -0.37           0.66           0.70          -0.26           0.77           0.81          -0.23

     -0.34           0.70           0.74           0.77           0.81           0.86          -0.19
  0.00    0.66  -0.37    0.70   0.66   -0.26   0.70    0.77  -0.26    0.81   0.77   -0.23   0.81    0.00
     -1.00          -0.37          -0.34          -1.00          -0.26          -0.23          -1.00

     -0.37           0.66           0.70          -0.26           0.77           0.81          -0.23
  0.00   -0.37  -1.00   -0.34  -0.37   -1.00  -0.34   -0.26  -1.00   -0.23  -0.26   -1.00  -0.23    0.00
      0.00           0.00           0.00           0.00           0.00           0.00           0.00
```

```
$ julia multi_armed_bandit.jl > mab.log
```
