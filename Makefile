.PHONY: fast perf tensorboard help clean

# Fast mode: Quick iteration (≤5 minutes)
# Default settings for rapid development and testing
fast:
	@echo "Starting fast training mode (≤5 minutes)..."
	python train_dqn_advanced.py \
		--device auto \
		--total_steps 50000 \
		--max_seconds 300 \
		--n_envs 1 \
		--batch_size 256 \
		--gradient_steps 2 \
		--n_step 1 \
		--train_freq 4 \
		--log_interval 1000 \
		--exp_name fast

# Performance mode: Heavier updates for better GPU utilization
# Opt-in mode with larger batch sizes and more gradient steps
perf:
	@echo "Starting performance training mode (opt-in)..."
	python train_dqn_advanced.py \
		--device auto \
		--total_steps 500000 \
		--max_seconds 3600 \
		--n_envs 1 \
		--batch_size 1024 \
		--gradient_steps 8 \
		--n_step 3 \
		--train_freq 4 \
		--log_interval 5000 \
		--use_amp \
		--exp_name perf

# Launch TensorBoard to view training metrics
tensorboard:
	@echo "Starting TensorBoard on http://localhost:6006"
	@echo "Press Ctrl+C to stop"
	tensorboard --logdir runs --port 6006

# Clean generated files (logs, checkpoints, etc.)
clean:
	@echo "Cleaning generated files..."
	rm -rf runs/
	rm -rf checkpoints/
	rm -rf wandb/
	rm -f *.log
	@echo "Clean complete!"

# Display help information
help:
	@echo "Snake RL - DQN Training Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make fast        - Fast training mode (≤5 min, batch_size=256, gradient_steps=2)"
	@echo "  make perf        - Performance mode (opt-in, batch_size=1024, gradient_steps=8, AMP enabled)"
	@echo "  make tensorboard - Launch TensorBoard to view training metrics"
	@echo "  make clean       - Remove generated files (runs/, checkpoints/, etc.)"
	@echo "  make help        - Display this help message"
	@echo ""
	@echo "Logs are saved to runs/<exp_name>/"
	@echo "  - CSV metrics: runs/<exp_name>/metrics.csv"
	@echo "  - TensorBoard: runs/<exp_name>/events.out.tfevents.*"
