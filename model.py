# model.py
import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from typing import List

# 1. 파라미터 초기화 함수
def init_network_params(key: jax.random.Key, sizes: List[int]) -> List:
    """신경망의 가중치(W)와 편향(b)을 초기화합니다."""
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    
    # Xavier/Glorot 초기화 스케일 팩터
    for in_dim, out_dim, layer_key in zip(sizes[:-1], sizes[1:], keys):
        scale = jnp.sqrt(6.0 / (in_dim + out_dim))
        W = jax.random.uniform(layer_key, shape=(in_dim, out_dim), minval=-scale, maxval=scale)
        b = jnp.zeros(out_dim)
        params.append((W, b))
    return params

# 2. 모델 적용(계산) 함수
def apply_model(params: List, x: jnp.ndarray) -> jnp.ndarray:
    """초기화된 파라미터로 순전파(forward pass)를 수행합니다."""
    # 마지막 레이어를 제외하고 활성화 함수 적용
    for W, b in params[:-1]:
        x = sigmoid(jnp.dot(x, W) + b)
    
    # 마지막 레이어
    final_W, final_b = params[-1]
    x = jnp.dot(x, final_W) + final_b
    
    return x