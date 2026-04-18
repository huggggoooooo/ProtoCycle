source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/miniconda3/envs/protein_chai
cd /path/to/ProtoCycle/eval_tools

# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/baseline_results/ground_truth.csv" --output_csv "/path/to/ProtoCycle/baseline_results/metrics/ground_truth_metrics_1.csv"

# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/baseline_results/Qwen2.5_7b_sft850_ep5.csv" --output_csv "/path/to/ProtoCycle/baseline_results/metrics/Qwen2.5_7b_sft850_ep5_metrics_1.csv"

# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/baseline_results/Qwen2.5_7b.csv" --output_csv "/path/to/ProtoCycle/baseline_results/metrics/Qwen2.5_7b_metrics.csv"

# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/baseline_results/Qwen3_8b.csv" --output_csv "/path/to/ProtoCycle/baseline_results/metrics/Qwen3_8b_metrics.csv"

# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/baseline_results/Qwen2.5_72B.csv" --output_csv "/path/to/ProtoCycle/baseline_results/metrics/Qwen3_2.5_72b_metrics.csv"

# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProDVa/prodva_chat_eval_CAMEO_100.csv" --output_csv "/path/to/ProtoCycle/baseline_results/metrics/ProDva_CAMEO_metrics.csv"

# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/baseline_results/pinal_CAMEO.csv" --output_csv "/path/to/ProtoCycle/baseline_results/metrics/Pinal_CAMEO_metrics.csv"

# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/baseline_results/Qwen2.5_7B_rl_step_10_CAMEO.csv" --output_csv "/path/to/ProtoCycle/baseline_results/metrics/Qwen2.5_7B_rl_step_10_CAMEO_metrics.csv"


# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProDVa/prodva_chat_eval_CAMEO_100.csv" --output_csv "/path/to/ProtoCycle/baseline_results/metrics/ProDva_CAMEO_metrics_infer_on_CAMEO.csv"

# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/desc2seq_agent_eval_clever_100_ci_1.csv" --output_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/results/ci_1_ground_truth.csv"

# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/desc2seq_agent_eval_clever_100_ci_2.csv" --output_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/results/ci_2_ground_truth.csv"

# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/results/pinal_ci_1.csv" --output_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/metrics/pinal_ci_1.csv"

# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/results/pinal_ci_2.csv" --output_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/metrics/pinal_ci_2.csv"

# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/results/prodva_ci_1.csv" --output_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/metrics/prodva_ci_1.csv"

# python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/results/prodva_ci_2.csv" --output_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/metrics/prodva_ci_2.csv"


python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/results/Qwen2.5_7B_sft_ci_1.csv" --output_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/metrics/Qwen2.5_7B_sft_ci_1_metrics.csv"

python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/results/Qwen2.5_7B_sft_ci_2.csv" --output_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/metrics/Qwen2.5_7B_sft_ci_2_metrics.csv"

python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/results/Qwen2.5_7B_rl_step20_ci_1.csv" --output_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/metrics/Qwen2.5_7B_rl_step20_ci_1_metrics.csv"

python compute_metrics_multi_gpu.py --input_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/results/Qwen2.5_7B_rl_step20_ci_2.csv" --output_csv "/path/to/ProtoCycle/ACL_exps/CI_exps/metrics/Qwen2.5_7B_rl_step20_ci_2_metrics.csv"