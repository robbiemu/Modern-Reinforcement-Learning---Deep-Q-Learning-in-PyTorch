Sección 2: Fundamentals of Reinforcement Learning
===

Agents, Environments, and Actions
---
### Agent
agents observe on state of env, acts on env, which cauases state change in ENV

agents correlates relations between states, actions, and rewards to learn good behavior

### Environment
positional state of content, and the corresponding rewards used in training (State really is the reading of the weight sensor)

### Actions 
actions cause changes in state

This is sort of a classification problem, but there is limited information... especially in continuous actions

Markov Decision Processes
---

### Decision processes
a decision process is sequence of states, actions and rewards.

each decision influences all feature rewards, as the availability of those rewards may be "blocked" or "unlocked" as a consequence (indicated at some point in state).

if the state depends purely on action and state that came before, this is a **Markov Devision Process** (MDP). Contrasting with multi-player games or games where state also mutates independently from itself and the agent. So, in MDP, actions cause state transitions.

### Maximizing rewards and episodic tasks

an agent maximizes its total reward over time. This series can be observed as an expected reward at some time t:
```
G_t = R_t+1+R_t+2...
```

### Terminal state

`G_t = 0` here, as this is a final state.

### Continuous tasks

`G_t` can now be infinite! So we discount rewards (following a power law to ensure it has a ceiling, with a modifier γ): `0 ≤ γ ≤ 1`. As γ approaches 1, rewards are minimized .. this produces "far-sighted" or strategic policy. As γ approaches 0, rewards are maximized (untouched) producing "myopic" or tactical policy.

_rule of thumb:_ `0.95 ≤ γ ≤ 0.99`

### Policy π

probablistic mapping of state to action (that can provide a future reward approximation).

In real terms, a policy is simply a mapping of state or state and action (leading to this location) to the next action. It is a single choice, for each state...

Value functions, action value functions, and the Bellman Equation
---

### Value function
since at each time step, a reward is calculated on the current state, and that state is determined (under MDP assumptions) by our actions, that also means that _state-action pairs_ have values. the value can be derrived explicitly given Expectations for reward and state, with the value function (see pdf).

### Action-Value function
the action to take at a given state-action pair (ie state, [given context]) is also derrivable explicitly given Expectations for reward and state. this is the action value function.
This is q(s,a) and this is the Q in Q learning

### Expected reward / state

we can estimate expected state by sampling results in interation with the environment. We average rewards give state / state,action pair. For large state spaces, we have to approximate the state space with NN models. 

The bellman equation is a relationship between temporal stages in state space.

### Rank ordering policies

we can use the bellman equation to derrive optimality for selections (policies) and therefore compose the set of optimal policies in the environment.

Model-free vs Model-based Learning
---

model-based methods of solving the bellman equation are such as dynamic programming, and require sufficient knowledge of all possible states to completely calculate the equation at the penultimate state and work backwards, thereby deducting the best policy set. this may not be possible.

model-free methods estimate the coefficients to produce a "best guess", thereby skipping the complexity of computing the entire state space and instead only calculating the next possibilities. Q-Learning is one such method. this is _trial-and-error_. This is also sometimes called model-estimation methods.

Explore-Exploit Dilemma
---

### optimistic initial values
agents will better explore if given small negative rewards.

### epsilon greedy
we can alternatively between random and remembered selections. I imagine this should work still with positive weights. epsilon must decrease over time, and must be finite. Usually between ε 0.01 and 0.1

Temporal Difference Learning
---

### refining the value function
each time the agent samples the state, it refines its model estimate on the environment, allowing it to improve its approximation of the true value function.

new_estimate = old_estimate + step size(target - old_estimate)

the target is in some proportion to the reward expected, if not that thing itself.

step_size controls rate of change, and can be dynamic as the agente progresses through its actions.

### episodic
in some forms of reinforcement learning, like Monte carlo algorithms, update the step_size at the end of each episode (after 1 complete game, per say)

### continuous
Q-Learning uses temporal difference: we update the estimate of the value function at each time step, rather than at each episode. This makes it easier to apply to continuous (never-ending) environments. this is a Bootstrapping approach.

remember that q in Q Learning is the action value function. the update equation is similar (so we can derrive it from the same coefficients and hyperparameters). The main difference is that we must select the action leading to maximum reward at each state
