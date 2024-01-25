## Quantum Annealing, config 1

Chain Strength Prefactor: 0.3
Annealing time: 200
$\lambda = max{W} * |V| + 1$

| Seq. Cnt. | Term. Cnt. | AC Rate | Opt. Sol |
|:---------:|:----------:|:-------:|:--------:|
|    10     |     4      | 2/10000 |    7     |
|    10     |     5      | 0/10000 |    -     |

## Quantum Annealing, config 2
Chain Strength Prefactor: 0.3
Annealing time: 200
$\lambda = max{W} * |V| + 1$
Lambda multiplier: `[7 9 4 2 2]`

| Seq. Cnt. | Term. Cnt. |  AC Rate  | Opt. Sol |
|:---------:|:----------:|:---------:|:--------:|
|    10     |     4      | 233/10000 |    7     |
|    10     |     5      |  2/10000  |    16    |

## Quantum Annealing, config 3
Chain Strength Prefactor: 0.3
Annealing time: 200
$\lambda = max{W} * |V| + 1$
Lambda multiplier: `[7 9 4 2 2]`
Long edge removed

| Seq. Cnt. | Term. Cnt. |  AC Rate  | Opt. Sol |
|:---------:|:----------:|:---------:|:--------:|
|    10     |     4      | 117/10000 |    7     |
|    10     |     5      |  2/10000  |    16    |

## Simulated Annealing

$\lambda = max{W} * |V| + 1$

| Seq. Cnt. | Term. Cnt. |  AC Rate   | Opt. Sol |
|:---------:|:----------:|:----------:|:--------:|
|    10     |     4      | 8797/10000 |    7     |
|    10     |     5      | 9019/10000 |    8     |

Note: Takes very long time to run. More than 1 hour for 10000 test with 5 terminals.

## Quantum annealing configs for 5 terminals

`[7 9 5 1 1]`: 0/1000
`[4 11 3 1 1]`: 1/10000
(0, 5)
(0, 6)
(4, 7)
(4, 8)
(5, 1)
(5, 4)
(6, 9)
(7, 2)
(7, 3)
`[4 11 4 1 1]`: 2/10000
(0, 5)
(0, 9)
(3, 6)
(3, 10)
(5, 3)
(5, 7)
(7, 4)
(7, 8)
(8, 2)
(9, 1)

## Optimal answers using ILP-SA
4: 7-7
5: 11-11
6: 13-14
7: 15-16

## Optimal answers, Quantum, improved 3rd constraint
### Config 1
`[2 7 5 1 1]`:
4: 34/1000