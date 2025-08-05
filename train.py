# train.py
import jax
import jax.numpy as jnp
import optax
from functools import partial

# 모델을 직접 import
from model import apply_model

# 옵티마이저 생성은 동일
def create_optimizer(config):
    scheduler = optax.piecewise_constant_schedule(
        init_value=config.learning_rate,
        boundaries_and_scales={
            milestone: config.lr_gamma for milestone in config.lr_milestones
        }
    )
    return optax.adam(learning_rate=scheduler)

# 손실 함수 (apply_fn 대신 apply_model 사용)
def compute_loss_fn(params, batch, config, problem, key):
    t0, x0 = batch

    def body_fn(n, val):
        key, t, x, total_loss = val
        loop_key, key = jax.random.split(key)
        
        model_input = jnp.concatenate([t, x], axis=-1)
        y = apply_model(params, model_input)

        grad_y_fn = jax.grad(lambda x_arg: apply_model(params, jnp.concatenate([t, x_arg], axis=-1)).sum())
        grad_y = jax.vmap(grad_y_fn)(x)
        
        # ... (이하 BSDE 시간 전개 로직은 이전과 거의 동일) ...
        # state.apply_fn({'params': p}, ...) 호출 부분을
        # apply_model(params, ...) 호출로 변경해주면 됩니다.
        
        # 예시:
        y_jumped = apply_model(params, jnp.concatenate([t, x_jumped], axis=-1))
        # ...
        y_next_nn = apply_model(params, jnp.concatenate([t_next, x_next], axis=-1))
        # ...
        
        # (이 부분은 설명을 위해 생략, 실제로는 이전 코드의 로직을 채워넣어야 함)
        step_loss = jnp.mean((y_next_nn - y_pred_next)**2) # y_pred_next는 BSDE 공식
        
        return (key, t_next, x_next, total_loss + step_loss)

    init_val = (key, t0, x0, 0.0)
    _, t_N, x_N, total_loss = jax.lax.fori_loop(0, config.n_steps, body_fn, init_val)
    
    g_target = problem.terminal_condition(x_N)
    y_N = apply_model(params, jnp.concatenate([t_N, x_N], axis=-1))
    terminal_loss = jnp.mean((y_N - g_target)**2)

    return total_loss + terminal_loss

# JIT 컴파일된 훈련 스텝
@partial(jax.jit, static_argnums=(4, 5))
def train_step(params, opt_state, batch, optimizer, config, problem, key):
    # value_and_grad에 들어가는 함수를 람다로 감싸서 params만 인자로 받도록 함
    loss_fn_for_grad = lambda p: compute_loss_fn(p, batch, config, problem, key)
    loss, grads = jax.value_and_grad(loss_fn_for_grad)(params)
    
    # 파라미터 업데이트
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss