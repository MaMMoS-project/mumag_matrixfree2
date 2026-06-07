This is the best part about using the stagnation method: **it is computationally free.**

You do not need to evaluate your actual function $f(x)$, and you do not need to do any extra matrix-vector multiplications to calculate the quadratic model $m_k$.

Because the inner loop is a Conjugate Gradient (CG) algorithm, it is already minimizing that exact quadratic model at every step. We can calculate the change in the model using only the variables you are *already* computing in Algorithm 7.1.

Here is the mathematical trick to do it.

### The Math: Why it's Free

The quadratic model is defined as:


$$m_k(z) = f_k + \nabla f_k^T z + \frac{1}{2} z^T B_k z$$

We want to know the decrease in this model between step $j$ and step $j+1$.


$$\text{Decrease} = m_k(z_j) - m_k(z_{j+1})$$

In your Algorithm 7.1, the update step is $z_{j+1} = z_j + \alpha_j d_j$. If we plug this into the model and use the mathematical properties of the CG algorithm (specifically, that the residual $r_j$ is exactly the gradient of the quadratic model, and $r_j^T d_j = -r_j^T r_j$), the messy algebra collapses into a beautifully simple formula:

$$m_k(z_j) - m_k(z_{j+1}) = \frac{1}{2} \alpha_j r_j^T r_j$$

Look at your Algorithm 7.1: **You already have both of those numbers.** $r_j^T r_j$ is the numerator of your $\alpha_j$ calculation!

### How to Implement It in Your Code

To use the stagnation condition: $m_k(z_j) - m_k(z_{j+1}) \le \nu (m_k(0) - m_k(z_j))$
You just need to track the **Total Reduction** ($Q$) from the start of the loop, and compare it to the **Current Step Reduction** ($dq$).

Here is exactly how you modify the inner loop of Algorithm 7.1 to include this (assuming $\nu = 10^{-3}$ and a hard cap of `j_max = 15`):

```text
Set z_0 = 0, r_0 = \nabla f_k, d_0 = -r_0
Set Q = 0.0  // Track the total model reduction

for j = 0, 1, 2, ..., j_max

    if d_j^T B_k d_j <= 0
        if j == 0
            return p_k = -\nabla f_k;
        else
            return p_k = z_j;

    // 1. Calculate standard CG step length
    Set r_sq = r_j^T r_j;               // You need this for alpha anyway
    Set \alpha_j = r_sq / (d_j^T B_k d_j);
    
    // 2. STAGNATION CHECK (New Code)
    Set dq = 0.5 * \alpha_j * r_sq;     // The decrease from this single step
    
    if j > 0 and dq <= \nu * Q          // Check if progress has stagnated
        return p_k = z_j;               // Exit early!
        
    Set Q = Q + dq;                     // Update total accumulated reduction

    // 3. Continue standard Algorithm 7.1 updates
    Set z_{j+1} = z_j + \alpha_j d_j;
    Set r_{j+1} = r_j + \alpha_j B_k d_j;
    
    // 4. RESIDUAL CHECK (Your modified eps_k)
    if ||r_{j+1}|| < \epsilon_k
        return p_k = z_{j+1};
        
    Set \beta_{j+1} = (r_{j+1}^T r_{j+1}) / r_sq;
    Set d_{j+1} = -r_{j+1} + \beta_{j+1} d_j;

end (for)

// 5. HARD CAP EXIT
return p_k = z_{j+1}; 

```

### Why this is perfect for your physics simulation

By tracking $Q$ and $dq$ this way, your inner solver becomes highly aware of its own usefulness.

* If a step results in a massive drop in the quadratic model (e.g., resolving a stiff short-range particle collision), $dq$ will be large, and the algorithm will keep going.
* Once the short-range forces are mostly balanced, $dq$ becomes a tiny fraction of the total work already done ($Q$), the `if dq <= \nu * Q` trips, and the algorithm safely exits without wasting a single unnecessary matrix operation.