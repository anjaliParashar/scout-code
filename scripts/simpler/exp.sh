
# BAMS
# python scripts/simpler/run_bams_simpler.py \
#     --logging-root   ./results_1/bams \
#     --scout-init-dir ./results/scout/init \
#     --pool-size 500 --n-grid 3 --seed 1 \
#     --n-rounds 20 --batch-size 3 --n-seeds 1 \
#     --lf-cost 0.8

# BNN-CV
python scripts/simpler/run_bnn_cv_simpler.py \
    --logging-root   ./results_2/bnn_cv \
    --scout-init-dir ./results_2/scout/init \
    --pool-size 1000 --n-grid 3 --seed 2 \
    --n-rounds 20 --batch-size 3 --n-seeds 1

python scripts/simpler/run_random_simpler.py     --logging-root   ./results_2/random     --scout-init-dir ./results_2/scout/init     --pool-size 1000 --n-grid 3 --seed 2     --n-rounds 20 --batch-size 3 --n-seeds 1

python scripts/simpler/run_gpc_simpler.py \
    --logging-root   ./results_2/gpc \
    --scout-init-dir ./results_2/scout/init \
    --pool-size 1000 --n-grid 3 --seed 2 \
    --n-rounds 20 --batch-size 3 --n-seeds 1 \
    --failure-threshold 1.0 --success-threshold 0.5
