# Portfolio-Optimizer
This project uses gradient descent and particle swarm algorithms to optimize portfolios to given user-defined parameters.

Utilizes numpy to efficiently and accurately carry out both algorithms. Given a list of individual investments and their parameters (alpha, beta, investment category), as well as optimization information including targets and weights, constructs a portfolio constructed from those investments designed to most closely align with the target values, prioritizing those with higher weights. If the number of investments involved is two, the project can also create animated graphics showing the algorithms at work.

## Features
- Calculates the ideal portfolio to match the given constraints using either or both algorithms.
- Displays side by side the statistics of the best fit portfolio to the parameters provided, as well as the raw error it achieved in doing so.
- Includes methods to calculate the alpha, beta, gradient, allocations, or error terms of all portfolios in an array of portfolios simultaneously
- For two dimensional data (two investment options), creates the following animated plots for gradient descent:
  - Raw pseudo-portfolio paths on a 2-dimensional heatmap of the error function (for more information on what is meant by 'pseudo-portfolio' in this context, see the section titled "Modifications to Portfolio Space due to gradient descent Bound Limitations")
  - True portfolio paths on a 2-dimensional heatmap of the error function
  - Raw pseudo-portfolio paths on a 3-dimensional wireplot surface of the error function
  - True portfolio paths on a 3-dimensional wireplot surface of the error function
- For two dimensional data (two investment options), creates the following animated plots for particle swarm:
  - Portfolio paths on a 2-dimensional heatmap of the error function
  - Portfolio paths on a 3-dimensional wireplot of the error function

## Methodology

### Notation & Variables
This section serves to describe the notation used for various variables and quantities throughout this document.

Indices:
- $i, k$: denotes the index of an investment in the list of investments
- $j, l$: denotes the index of a category in the list of allocation categories

Index Counts:
- $N$: the number of investment options; the length of all vectors indexed by $i$ or $k$
- $M$: the number of category options; the length of all vectors indexed by $j$ or $l$

i-Indexed Vectors:
- $p$: a portfolio vector, where each entry, $p_i$, denotes the percentage of $p$ that is invested in investment $i$. Such vectors must represent valid portfolios, so their entries must be nonnegative and add up to 1.
- $\tilde{p}$: a pseudo-portfolio vector, as defined in the gradient descent section. These vectors can include negative values and don't have to add up to 1, but they cannot be zero vectors.
- $α$: the vector of alpha values for each investment, $α_i$ denotes the alpha of investment $i$
- $β$: the vector of beta values for each investment, $β_i$ denotes the beta of investment $i$
- $b_p$: the best portfolio vector achieved by particle $p$ so far when executing particle swarm optimization
- $b_G$: the best global portfolio vector achieved by all particles so far when executing particle swarm optimization
- $v_p$: the velocity of portfolio vector $p$

j-Indexed Vectors:
- $a_p$: the allocation vector corresponding to portfolio $p$, where each entry, $a_{j, p}$ denotes the percentage of portfolio $p$ invested in category $j$.

Matrices:
- $a$: the matrix of allocation values for an investment category, where $a_j$ is the vector of allocation values for a single investment category j, and $a_{j, i}$ is 1 if investment $i$ is of category $j$, and 0 otherwise

Portfolio Statistics:
- $α_p$: the alpha of portfolio $p$
- $β_p$: the beta of portfolio $p$

Errors:
- $E$: the total error of a portfolio
- $E_α$: the alpha component of the error of a portfolio, before weights are applied
- $E_β$: the beta component of the error of a portfolio, before weights are applied
- $E_a$: the allocation component of the error of a portfolio, before weights are applied

User-defined values used for both optimization techniques:
- $β_T$: the target beta value
- $a_T$: the vector of allocation targets, where $a_{j, T}$ is the target allocation in category $j$
- $W_α$: the weight given to the alpha component of the error
- $W_β$: the weight given to the beta component of the error
- $W_a$: the weight given to the allocation component of the error

User-defined values for gradient descent optimization:
- $L$: the learning rate

User-defined values for particle swarm optimization
- $I$: the inertia coefficient (how much a particle's motion from the last timestep carries over into the current timestep)
- $S$: the social coefficient (how quickly a particle moves towards the global minimum $b_G$)
- $C$: the cognitive coefficient (how quickly a particle $p$ moves towards its personal best, $b_p$)
- $R$: the random coefficient (how much of a particle's movement is randomly determined)

### Calculating Portfolio alpha, beta, and allocations

Portfolio alpha, beta, and allocations are all calculated based on weighted sums of investment alpha, beta, and allocation category, respectively.

```math
\begin{aligned}
α_p &= p \cdot α = \sum p_i α_i\\
β_p &= p \cdot β = \sum p_i β_i\\
a_{j, p} &= p \cdot a_j = \sum p_i a_{j, i}
\end{aligned}
```

### Error Function
The error function for this implementation is the weighted sum of error functions for alpha, beta, and allocation alignment:
```math
\begin{aligned}
E = W_α E_α + W_β E_β + W_a E_a
\end{aligned}
```

Since a larger alpha is universally desirable, there is no target alpha. Instead, alpha error of a portfolio i is defined as:
```math
\begin{aligned}
E_α = e^{-α_p}
\end{aligned}
```

Given a target beta of $β_T$, the error associated with beta for a given portfolio i is given as the squared difference between its own beta and $β_T$:
```math
\begin{aligned}
E_β = (β_p - β_T)^2
\end{aligned}
```

Allocation error can be thought of as a collection of M different category errors. For each such category, the category error of a portfolio is the squared difference between its own allocations within that category and the target allocations in that category:
```math
\begin{aligned}
E_a = \sum (a_{j, p} - a_{j, T})^2
\end{aligned}
```

### Gradient Descent
In traditional gradient descent, one initial vector v is chosen, and then updated at each time step by moving it opposite the gradient of the error function at that point, scaled by a specified learning rate. In this implementation, any number of initial vectors may be created. If so, all vectors will be updated simultaneously, and the result that has the smallest error will be selected for the final returned portfolio.

The gradient of the error function is a vector pointing in the direction of steepest ascent, and has magnitude determined by the speed of ascent. As such, so long as the error function is smooth, this method tends to move vectors to positions with a lower error function, with a speed proportional to the expected decrease in error at each step.

As such, the update at each timestep would be:
$v \rightarrow v - L \cdot ∇E|_v$

The algorithm then continues for a predefined number of timesteps, and returns the resultant vector after that many timesteps.

#### Modifications to Portfolio Space due to Gradient Descent Bound Limitations
As mentioned above, a vector is feasible if and only if it has nonnegative values that add up to 1. This creates a domain equivalent to the first quadrant of the L1 unit sphere. This presents a challenge for standard gradient descent, as the algorithm tends to fail at the boundaries of enclosed domains. This is due to the fact that if the gradient vector were to point inwards at the boundary, the method would attempt to move points in the opposite direction, outside the boundary. It is for this reason that this implementation utilizes an alternative portfolio vector space where gradient descent can operate without any issues. That space is referred throughout this readme as "pseudo portfolio space" and vectors within it are referred to as "pseudo portfolios", as opposed to feasible portfolio vectors "true portfolios."

Pseudo portfolio space takes any non-zero N dimensional vector to be valid except the zero vector, rather than just those satisfying the above feasibility requirements. Each pseudo portfolio \tilde{p} corresponds to a true portfolio p, which can be found using the following mapping:
```math
\begin{aligned}
p_i = \frac{{\tilde{p}^2_i}}{||\tilde{p}||^2}
\end{aligned}
```
Where $||{\tilde{p}}|| = (\sum \tilde{p}_i^2)^{1/2}$ is the L2 norm of $\tilde{p}$.

The error for pseudo portfolios can then be defined as the error for its corresponding true portfolio. In fact, all parameters for $\tilde{p}$ can be defined as the same as $p$:
$α_\tilde{p} = α_p$, $β_\tilde{p} = β_p$, etc.

Since this implementation of gradient descent works on pseudo portfolio space, the update for each timestep becomes:
```math
\begin{aligned}
\tilde{p} \rightarrow \tilde{p} - L \cdot ∇E|_\tilde{p}
\end{aligned}
```
With ∇ the gradient in pseudo-portfolio space.

When initializing the optimization algorithm, the particles in pseudo portfolio space are sampled randomly within the annulus starting at distance 0.2 from the origin and extending to a distance of 1 from the origin.

Note:
- Under this mapping, each $p_i$ is certainly nonnegative, and $\sum p_i = \sum (\tilde{p}_i^2 / ||\tilde{p}||^2) = \sum (\tilde{p}_i^2) / ||\tilde{p}||^2 = 1$, showing that $p$ is indeed a feasible portfolio.
- This is not a one to one mapping, as multiple pseudo-portfolios can map to the same true portfolio (e.g. $\tilde{p}$ and $c\tilde{p}$ both map to the same $p$ for any nonzero scalar $c$)
- For $\tilde{p} = 0$, $||\tilde{p}|| = 0$, which breaks the mapping. This is why $\tilde{p}$ cannot be the zero vector.
- This mapping is smooth except at the origin. Because rescaling vectors $\tilde{p}$ does not change the corresponding true portfolio $p$, the error $E_p$ remains constant for rescaling as well. This forces the gradient vector to be perpendicular to the vector $\tilde{p}$, meaning it will never point in towards the origin. As such, portfolios will never be drawn towards the discontinuity, avoiding any problems it might present.

#### Calculating the Gradient
Before calculating the gradient, $α_p$, $β_p$, and $a_{j, p}$ need to be converted to pseudo portfolio space.

Using $p_i = \tilde{p}_i^2 / ||\tilde{p}||^2$ as above:
```math
\begin{aligned}
α_\tilde{p} = α_p = \sum p_i α_i = \sum \frac{\tilde{p}_i^2 α_i }{||\tilde{p}||^2} = \frac{\sum \tilde{p}_i^2 α_i}{||\tilde{p}||^2}
\end{aligned}
```
And since $β_p$ and $a_{j, p}$ are defined similarly:
```math
\begin{aligned}
β_\tilde{p} = β_p = \frac{\sum \tilde{p}_i^2 β_i}{||\tilde{p}||^2}\\
a_{j, \tilde{p}} = a_{j, p} = \frac{\sum \tilde{p}_i^2 a_{j, i}}{||\tilde{p}||^2}
\end{aligned}
```

The gradient of the error function is then:
```math
\begin{aligned}
∇E = ∇(W_α E_α + W_β E_β + W_a E_a) = W_α ∇E_α + W_β ∇E_β + W_a ∇E_a
\end{aligned}
```
##### Calculating ∇E_α
$∇E_α$ can be calculated by finding the partial derivatives:
```math
\begin{aligned}
\frac{∂}{∂\tilde{p}_i} E_α &= \frac{∂}{∂\tilde{p}_i}\left(e^{-α_p}\right) = -\frac{∂}{∂\tilde{p}_i}(α_p)e^{-α_p}
\end{aligned}
```

And then, calculating $∂/∂\tilde{p}_i(α_p)$:
```math
\begin{aligned}
\frac{∂}{∂\tilde{p}_i}(α_p) &= \frac{∂}{∂\tilde{p}_i}\left(\frac{\sum \tilde{p}_k^2 α_k}{||\tilde{p}||^2}\right)\\
&= \left(\frac{||\tilde{p}||^2\frac{∂}{∂\tilde{p}_i}\left(\sum \tilde{p}_k^2 α_k\right) - \left(\sum \tilde{p}_k^2 α_k\right)\frac{∂}{∂\tilde{p}_i}\left(\sum \tilde{p}_k^2\right)}{||\tilde{p}||^4}\right)\\
&= \left(\frac{2||\tilde{p}||^2\tilde{p}_i α_i - \left(\sum \tilde{p}_k^2 α_k\right)(2\tilde{p}_i)}{||\tilde{p}||^4}\right)\\
&= \left(\frac{2||\tilde{p}||^2\tilde{p}_i α_i - ||\tilde{p}||^2α_p(2\tilde{p}_i)}{||\tilde{p}||^4}\right)\\
&= \frac{2\tilde{p}_i}{||\tilde{p}||^2}( α_i - α_p)
\end{aligned}
```

With this, the value of $∂/∂\tilde{p}_i$(E_α) becomes:
```math
\begin{aligned}
\frac{∂}{∂\tilde{p}_i} E_α = \frac{-2\tilde{p}_ie^{-α_p}}{||\tilde{p}||^2}( α_i - α_p)
\end{aligned}
```
From this, $∇E_α$ is then:
```math
\begin{aligned}
∇E_α = \left(\frac{-2e^{-α_p}}{||\tilde{p}||^2}\right)\tilde{p}\odot( α - α_p\mathbf{1})
\end{aligned}
```
Where $\mathbf{1}$ is the $N$ dimensional vector where every entry is 1.

##### Calculating ∇E_β
The same method can also be used to solve for $∇E_β$:
```math
\begin{aligned}
\frac{∂}{∂\tilde{p}_i} E_β &= \frac{∂}{∂\tilde{p}_i}(β_p - β_T)^2 = 2(β_p - β_T)\frac{∂}{∂\tilde{p}_i}(β_p)
\end{aligned}
```
Since $β_p = \sum \tilde{p}_i^2 β_i/||\tilde{p}||^2$ and $α_p = \sum \tilde{p}_i^2 α_i/||\tilde{p}||^2$ have exactly the same structure:
```math
\begin{aligned}
\frac{∂}{∂\tilde{p}_i}(β_p) = \frac{2\tilde{p}_i}{||\tilde{p}||^2}( β_i - β_p)
\end{aligned}
```

Plugging this back in to $∂/∂\tilde{p}_i E_β$:
```math
\begin{aligned}
\frac{∂}{∂\tilde{p}_i} E_β &= 2(β_p - β_T)\left(\frac{2\tilde{p}_i}{||\tilde{p}||^2}( β_i - β_p)\right)\\
&= \frac{4(β_p - β_T)}{||\tilde{p}||^2}\tilde{p}_i( β_i - β_p)
\end{aligned}
```
This gives $∇E_β$ to be:
```math
\begin{aligned}
∇E_β = \frac{4(β_p - β_T)}{||\tilde{p}||^2}\tilde{p}\odot( β - β_p\mathbf{1})
\end{aligned}
```

##### Calculating ∇E_a
Starting in the same manner as the previous error terms:
```math
\begin{aligned}
\frac{∂}{∂\tilde{p}_i} E_a &= \frac{∂}{∂\tilde{p}_i}\sum_j (a_{j, p} - a_{j, T})^2 = \sum_j 2(a_{j, p} - a_{j, T})\frac{∂}{∂\tilde{p}_i}(a_{j, p})
\end{aligned}
```

Again, since $a_{j, p}$ shares the same structure as $β_p$ and $α_p$, the derivation is the same:
```math
\begin{aligned}
\frac{∂}{∂\tilde{p}_i}(a_{j, p}) = \frac{2\tilde{p}_i}{||\tilde{p}||^2}( a_{j, i} - a_{j, p})
\end{aligned}
```

Plugging this back in:
```math
\begin{aligned}
\frac{∂}{∂\tilde{p}_i} E_a &= \sum_j \frac{4(a_{j, p} - a_{j, T})\tilde{p}_i}{||\tilde{p}||^2}( a_{j, i} - a_{j, p})\\
&= \frac{4\tilde{p}_i}{||\tilde{p}||^2}\sum_j (a_{j, p} - a_{j, T})( a_{j, i} - a_{j, p})\\
&= \frac{4\tilde{p}_i}{||\tilde{p}||^2}\left[\sum_j (a_{j, p} - a_{j, T})a_{j, i} - \sum_j (a_{j, p} - a_{j, T})a_{j, p}\right]
\end{aligned}
```

And as such:
```math
\begin{aligned}
∇E_{a} &= \frac{4}{||\tilde{p}||^2}\tilde{p}\odot \left[a^T(a_{p} - a_{T}) - ((a_p - a_T)\cdot a_p)\mathbf{1}\right]
\end{aligned}
```

##### Calculating ∇E
Putting all the pieces from previous calculations together:
```math
\begin{aligned}
∇E &= W_α ∇E_α + W_β ∇E_β + W_a ∇E_a\\
&=W_α\left(\frac{-2e^{-α_p}}{||\tilde{p}||^2}\right)\tilde{p}\odot( α - α_p\mathbf{1}) + W_β \left(\frac{4(β_p - β_T)}{||\tilde{p}||^2}\right)\tilde{p}\odot( β - β_p\mathbf{1}) + W_a\left(\frac{4}{||\tilde{p}||^2}\tilde{p}\right)\odot \left[a^T(a_{p} - a_{T}) - ((a_p - a_T)\cdot a_p)\mathbf{1}\right]\\
&= \frac{2}{||\tilde{p}||^2}\tilde{p}\odot\left( -W_αe^{-α_p}( α - α_p\mathbf{1}) + 2W_β(β_p - β_T)( β - β_p\mathbf{1}) + 2W_a(a^T(a_{p} - a_{T})- ((a_p - a_T)\cdot a_p)\mathbf{1})\right)
\end{aligned}
```

Note that because this only depends on $\tilde{p}$ in the term $2/||\tilde{p}||^2\tilde{p}$, and since scaling $\tilde{p}$ does not change its corresponding $p$ value, scaling $\tilde{p}$ becomes equivalent to changing the learning rate for that particular $\tilde{p}$ only. As such, by having multiple initial $\tilde{p}$ values with varying magnitudes, the model essentially is able to use multiple learning rates simultaneously and find the best fit among all such points.

### Particle Swarm
Particle swarm optimization works in a similar manner to gradient descent optimization, though it inherently must use a population of vectors. Instead of using the gradient to update timesteps, particle swarm optimization stores the best locations found by each vector in the population so far, as well as the best global location found across all vectors so far. Since it does not rely on the gradient, non-smooth surfaces work perfectly fine with particle swarm optimization, and as such, the entire calculation may be done entirely in real portfolio space, rather than requiring the same pseudo portfolio workaround that gradient descent requires.

Starting with a population of particles chosen (uniformly) randomly within valid portfolio space, particle swarm optimization first finds the best location among all such particles. Then, at the update step, it calculates the new velocity for each particle:
```math
\begin{aligned}
v_p \rightarrow Iv_p + r_1 S (b_g - p) + r_2 C (b_p - p) + R r_p
\end{aligned}
```
Where $r_1$ and $r_2$ are random floating point numbers between 0 and 1 (re-randomized each timestep), and $r_p$ is a random portfolio vector with all values between -1 and 1 (which need not satisfy the requirements to be a true portfolio, and is also re-randomized at each timestep).

The update is then:
```math
\begin{aligned}
p\rightarrow p + v_p
\end{aligned}
```

After each timestep, the updated $p$ vector may no longer be a valid portfolio, so an additional step is required:

```math
\begin{aligned}
p_i \rightarrow \frac{|p_i|}{\sum|p_k|}
\end{aligned}
```

Finally, each new position is analyzed to update $b_p$ and $b_g$ with any new best positions.

With this method of updating particles, particles are naturally drawn towards known global and local minima of the error function. However, one of the limitations of this method is that for sufficiently large N, it may take considerably more particles to sufficiently explore portfolio space and find the true global minimum.

## Example Output
### Particle Swarm

#### Figures
<img width="1600" height="600" alt="PS report" src="https://github.com/user-attachments/assets/4ae6fe69-39b8-4970-946c-08c8b701819f" />

#### Text Output
-=Portfolio=-
44.1% ABC, 55.9% DEF

Allocations:
44.1% Large Value, 55.9% Large Growth

Desired Allocations:
40.0% Large Value, 60.0% Large Growth

MPT Stats:
Alpha: -0.07050849194225402, Beta: 0.9353898096692951

Desired Beta: 0.8

Error: 0.3619416915691168

### Gradient Descent

#### Figures
<img width="1200" height="1000" alt="GD report" src="https://github.com/user-attachments/assets/76e6fc39-8b7d-4371-a39e-2260b503f9c0" />

#### Text Output
-=Portfolio=-
44.1% ABC, 55.9% DEF

Allocations:
44.1% Large Value, 55.9% Large Growth

Desired Allocations:
40.0% Large Value, 60.0% Large Growth

MPT Stats:
Alpha: -0.07050030020816855, Beta: 0.9353996397501977

Desired Beta: 0.8

Error: 0.3619416908282187


## Input Data
This implementation takes a text file as its input. The first non-empty line must be the weights $W_a$, $W_α$, $W_β$, and target beta, $β_T$, as comma separated values. The second non-empty line must be the desired allocations, formatted as:

$A_1: a_{1, T}, A_2: a_{2, T}, ..., A_M: a_{M, T}$, where $A_j$ is the name of allocation category $j$.

Any subsequent lines must define an investment, following the format:
$I_i: α_i, β_i, A_i$ where $I_i$ is the name of the ith investment option, and $A_i$ denotes the name allocation category to which the ith investment option belongs.

All spaces and empty lines are ignored, and any line starting with # will be ignored.

## Assumptions and Limitations

### Portfolio Model
- Each investment may only belong to a single investment allocation category, and is assumed to be 100% invested in that category.
- Alpha, beta, and category allocations for investment options are assumed to be fixed.
- Portfolio quality is defined by the error function
  - Weights may have to be tweaked or altered by the user to find a useful error function
- The alpha error function is exponential, while the beta and category errors are difference squared. This can mean that for significant differences in alpha between investment options, alpha evaluation may overpower the other parameters.
- Target category allocations are assumed to sum to 1

### Gradient Descent Optimization
- Gradient descent optimization is not guaranteed to find the best global minimum of the error function; it may find a local minimum instead.
- A suitable learning rate $L$ must be established.
- Requires a differentiable/smooth error function (which is satisfied in this case)
- May struggle near flat locations of the error function, where the gradient is nearly zero.
- Sensitive to initial location
- Needs only a single particle to operate effectively; multiple particles merely help mitigate initial location sensitivity

### Particle Swarm Optimization
- Particle swarm optimization is not guaranteed to find the best global minimum of the error function; premature convergence to non-optimal regions is possible
- Suitable inertia coefficient, cognitive coefficient, social coefficient, and random coefficient must be established.
- For large N, particle swarm optimization may require significantly more particles to adequately function.
- Sensitive to initial particle distribution
- Requires evaluating multiple particles per iterations

## Usage Example

### Input File (Two Investments)
```
#allocation weight, alpha weight, beta weight, target beta
1, 0.3, 2, 0.8

#desired allocations
Large Value: 0.4, Large Growth: 0.6

#investments to consider (name, alpha, beta, category)
ABC, -0.35, 0.6, Large Value
DEF, 0.15, 1.2, Large Growth
```

### Input Code (Two Investments)
```python
from PortfolioModel import PortfolioModel
from Population import GradDesc, ParticleSwarm

model = PortfolioModel('example_data.txt')

GD = GradDesc(model, 20, 0.01)

PS = ParticleSwarm(model, 20, 0.2, 0.1, 0.2, 0.05)

GD.draw('GD report.gif', 50)
PS.draw('PS report.gif', 50)
```

### Input File (Many Investments)
```
#allocation weight, alpha weight, beta weight, target beta
1, 0.3, 2, 0.8

#desired allocations
Large Value: 0.2, Large Growth: 0.4, Mid Blend: 0.1, Bond: 0.2, Small Growth: 0.1

#investments to consider (name, alpha, beta, category)
ABC, -0.35, 0.6, Large Value
DEF, 0.15, 1.2, Large Growth
GHI, 0.2, 0.9, Small Growth
JKL, -0.3, 0.8, Small Growth
MNO, -0.7, 0.7, Mid Blend
PQR, -1.2, 0.2, Bond
```
### Input Code (Many Investments)
```python
from PortfolioModel import PortfolioModel
from Population import GradDesc, ParticleSwarm

model = PortfolioModel('example_data.txt')

GD = GradDesc(model, 20, 0.01)

PS = ParticleSwarm(model, 20, 0.2, 0.1, 0.2, 0.05)

GD.simulate(50)
PS.simulate(50)

GD.report()
print()
PS.report()
```
