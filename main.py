# main.py
import jax
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import PIDEConfig
from model import init_network_params  # init 함수 import
from problems import HighDimPIDE
from train import create_optimizer, train_step

def run_training():
    config = PIDEConfig()
    problem = HighDimPIDE(config)
    
    key = jax.random.key(0)
    model_key, train_key = jax.random.split(key)

    # 1. 모델 파라미터 직접 초기화
    layer_sizes = [config.dim + 1] + list(config.hidden_dims)
    params = init_network_params(model_key, layer_sizes)

    # 2. 옵티마이저 및 상태 초기화
    optimizer = create_optimizer(config)
    opt_state = optimizer.init(params)

    loss_list = []
    print("Training Started with Pure JAX...")
    
    pbar = tqdm(range(1, config.epochs + 1))
    for epoch in pbar:
        epoch_key, train_key = jax.random.split(train_key)

        batch = problem.sample_initial_conditions(epoch_key, config.batch_size)
        
        # 3. params와 opt_state를 직접 관리
        params, opt_state, loss = train_step(params, opt_state, batch, optimizer, config, problem, epoch_key)
        loss_list.append(loss.item())

        if epoch % 1000 == 0:
            pbar.set_description(f"EPOCH: {epoch}/{config.epochs} | Loss: {loss.item():.6f}")

    print("Training Finished.")
    
    # --- 결과 시각화 및 평가 (이하 동일) ---
    # ...

if __name__ == '__main__':
    run_training()