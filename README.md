# llmrl: Composable Reinforcement Learning for Language Models

A flexible framework for training LLMs using various Reinforcement Learning techniques, with a focus on composable and customizable loss functions. The loss function is the main object here, which allows you to design your own RL scheme supporting combinations of reward signals as well as more complex settings like multi turn RL.

Main features:

- **Composable Loss Functions**: Build complex training objectives by combining different loss components
- **RL algorihtms through loss componetns**: Plan is to support loss components for various algorithms, including:
  - PPO
  - GRPO
  - Muesli-style policy improvement
  - MuZero-inspired consistency losses
- **Multiple/complex Reward Functions**: Composability provides a way to integrate multiple reward signals into the mix.


## Example

The goal of this project is to be able to do things like this:

```
grpo_loss_1 = loss.GroupedPolicyGradientLoss(
        name="policy_gradient_1",
        model=model,
        reward_function=reward_fun_1,
        clip_ratio=0.2, 
        normalize_advantages=True
    )

grpo_loss_2 = loss.GroupedPolicyGradientLoss(
        name="policy_gradient_2",
        model=model,
        reward_function=reward_fun_2,
        clip_ratio=0.2, 
        normalize_advantages=True
    )

entropy_reg = loss.EntropyRegularizer(
    name="entropy_regularizer",
    model=model,
    coefficient=0.01
)
kl_reg = loss.KLDivergenceRegularizer(
    name="kl_div_regularizer",
    model=model,
    coefficient=0.1, 
    target_kl=0.01, 
    adaptive=True
)

loss_fn = loss.CompositeLoss() + grpo_loss_1 + grpo_loss_2 + entropy_reg + kl_reg

trainer = Trainer(
    dataset=dataset,
    composite_loss=loss_fn,
    optimizer_kwargs={"learning_rate": 1e-4}
)


trainer.train(
        num_epochs=10,
)
```




